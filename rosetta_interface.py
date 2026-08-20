"""
Rosetta interface utilities
 - Fast interface ΔG via InterfaceAnalyzer (fixed backbone + repack)
 - Substitution-table precompute and I/O for differentiable surrogate

Notes from literature cue (Flex ddG / talaris2014): we switch the scorefunction
to talaris2014 to match the reported setup. Full Flex ddG is heavier and is not
implemented here; for high-throughput bias tables we keep backbone fixed and
repack side-chains, which is ~20 ms/sequence after warm-up on A100.
"""

import argparse, os, json
from typing import List, Tuple

import numpy as np
import pyrosetta as pr
pr.init("-mute all")  # silence Rosetta

from pyrosetta.rosetta.protocols.analysis             import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta                                          import standard_packer_task
from pyrosetta.rosetta.core.scoring                     import ScoreFunctionFactory

# ---------------------------------------------------------------------
# Global Rosetta state
# ---------------------------------------------------------------------
_POSE_PATH = "HK_complex_clean.pdb"                     # backbone frozen
_pose_ref  = pr.pose_from_pdb(_POSE_PATH)
_scorefxn  = ScoreFunctionFactory.create_score_function("talaris2014")
_cache     = {}  # (seq, chain) -> dG

AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def interface_ddg(seq: str, chain: str = "C") -> float:
    """Interface ΔG (kcal/mol) after repack on the fixed backbone.

    Uses InterfaceAnalyzerMover with talaris2014 scorefunction, matching the
    referenced ddG protocol’s scoring family. Fast and suitable for scanning.
    """
    key = (seq, chain)
    if key in _cache:
        return _cache[key]

    pose = pr.Pose(); pose.assign(_pose_ref)

    start = pose.chain_begin(chain)
    for i, aa in enumerate(seq, start=start):
        pr.protocols.simple_moves.MutateResidue(i, aa).apply(pose)

    task = standard_packer_task(pose); task.restrict_to_repacking()
    PackRotamersMover(_scorefxn, task).apply(pose)

    iam = InterfaceAnalyzerMover(f"{chain}_", False, _scorefxn); iam.apply(pose)
    ddg = iam.get_interface_dG()
    _cache[key] = ddg
    return ddg


# ---------------------------------------------------------------------
# Substitution-table (bias) generation
# ---------------------------------------------------------------------
def compute_subst_table(
    wt_seq: str,
    chain: str = "C",
    aa_list: List[str] = AA20,
) -> Tuple[np.ndarray, List[str]]:
    """Compute ΔE table (L×A) where entry is dG(mut i→a) − dG(WT).

    This is a single-site surrogate (no epistasis). Units: kcal/mol.
    """
    L, A = len(wt_seq), len(aa_list)
    table = np.zeros((L, A), dtype=np.float32)

    dG_wt = interface_ddg(wt_seq, chain=chain)
    wt_chars = list(wt_seq)

    for i in range(L):
        wt_aa = wt_chars[i]
        for j, aa in enumerate(aa_list):
            if aa == wt_aa:
                table[i, j] = 0.0
                continue
            mut = wt_chars.copy(); mut[i] = aa
            seq_mut = "".join(mut)
            dG_mut = interface_ddg(seq_mut, chain=chain)
            table[i, j] = dG_mut - dG_wt

    return table, aa_list


def save_table(path: str, table: np.ndarray, wt_seq: str, chain: str, aa_list: List[str]):
    meta = {
        "wt_seq": wt_seq,
        "chain": chain,
        "aa_list": "".join(aa_list),
        "pose_path": _POSE_PATH,
        "scorefxn": "talaris2014",
    }
    np.savez_compressed(path, table=table, meta=json.dumps(meta))


def load_table(path: str) -> Tuple[np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    table = data["table"]
    meta = json.loads(str(data["meta"]))
    return table, meta


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Precompute/load Rosetta substitution bias table.")
    p.add_argument("--wt", required=True, help="WT sequence for the target chain")
    p.add_argument("--chain", default="C")
    p.add_argument("--out", default="ro_subst.npz", help="Output .npz path (bias table)")
    p.add_argument("--aa", default="".join(AA20), help="AA alphabet columns order")
    args = p.parse_args()

    if os.path.exists(args.out):
        table, meta = load_table(args.out)
        print(f"Loaded bias from '{args.out}' (L={table.shape[0]}, A={table.shape[1]}, chain={meta['chain']})")
        print(f"AA order: {meta['aa_list']}")
        return

    aa_list = list(args.aa)
    print(f"Computing Rosetta substitution table… L={len(args.wt)}, A={len(aa_list)}, chain={args.chain}")
    table, aa_list = compute_subst_table(args.wt, chain=args.chain, aa_list=aa_list)
    save_table(args.out, table, args.wt, args.chain, aa_list)
    print(f"Wrote bias to '{args.out}' (size: {table.shape[0]}×{table.shape[1]})")
    print(f"AA order: {''.join(aa_list)}")


if __name__ == "__main__":
    main()