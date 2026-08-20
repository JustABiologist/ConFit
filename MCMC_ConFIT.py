"""
DMALAProteinSampler + Langevin search driver
===========================================
This file now contains **two main components**:

1. **`DMALAProteinSampler`** – the gradient-based discrete Langevin sampler we
   added earlier, *unchanged* except that its private ``_energy`` method has been
   specialised for *masked-LM* ESM models (negative log-likelihood over all
   positions).
2. **`LangevinSearch`** – a wrapper that mirrors the previous simulated-annealing
   interface but uses the DMALA sampler under the hood.  It supports multiple
   parallel chains, keeps track of best sequences, and exposes nearly the same
   run-time CLI flags.

The old simulated-annealing class (`SequenceMCMC`) is still here for reference
and can be selected via ``--sampler sa``.  The new Langevin path is chosen with
``--sampler langevin`` (default).

> **NOTE**: The gradient path requires a *differentiable* energy.  We define it
> as the *negative* average log-probability under the MLM (no masking).  This is
> empirical but works well in practice and keeps gradients alive.
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import argparse, textwrap, random, os, sys, json
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import EsmForMaskedLM, EsmTokenizer
from peft import PeftModel
from accelerate import Accelerator

# ---------------------------------------------------------------------
# DMALAProteinSampler (unchanged, but with MLM-aware energy)
# ---------------------------------------------------------------------

class DMALAProteinSampler(torch.nn.Module):
    """Discrete Metropolis-Adjusted Langevin sampler.

    * keeps residues [0 … prefix-1] frozen
    * clips ∂U/∂x to ±grad_clip
    """

    def __init__(
        self,
        seq_len: int,
        aa_token_ids: torch.LongTensor,
        *,
        step_size: float = 0.25,
        temp: float = 1.0,
        preconditioner: torch.Tensor | None = None,
        n_steps: int = 1,
        decay: float = 0.99,
        grad_clip: float = 10.0,
        constant_prefix: int = 20,
        max_flips: int = 0,          # 0 ⇒ no cap; else ≤ k flips per sweep
    ):
        super().__init__()
        self.L, self.A = seq_len, aa_token_ids.numel()
        self.ids = aa_token_ids
        self.alpha, self.T = step_size, temp
        self.n_steps, self.decay = n_steps, decay
        self.grad_clip, self.prefix = grad_clip, constant_prefix
        self.k = max_flips

        g = torch.ones(seq_len) if preconditioner is None else preconditioner.float()
        assert g.numel() == seq_len and (g > 0).all()
        self.register_buffer("g", g)
        self.register_buffer("acc_rate", torch.tensor(0.0))

    # ------------------------------------------------ helpers
    def _one_hot(self, x: torch.LongTensor) -> torch.FloatTensor:
        return F.one_hot(x, num_classes=self.A).float()

    def _energy(self, one_hot: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """-log P averaged over positions (FP32, no masking)."""
        with torch.cuda.amp.autocast(enabled=False):
            B, L, _ = one_hot.shape
            W = model.get_input_embeddings().weight                 # (V,d)
            emb = torch.einsum("bla,ad->bld", one_hot, W[self.ids]) # (B,L,d)

            bos_id = model.config.bos_token_id or 1
            eos_id = model.config.eos_token_id or 2
            emb = torch.cat([W[bos_id].expand(B,1,-1),
                             emb,
                             W[eos_id].expand(B,1,-1)], dim=1)      # (B,L+2,d)

            # add positional embeddings
            pos = model.esm.embeddings.position_embeddings.weight[:L+2]
            emb = emb + pos.unsqueeze(0).to(emb.dtype)

            hidden = model.esm.encoder(emb, return_dict=True).last_hidden_state
            logits = model.lm_head(hidden).log_softmax(-1)          # (B,L+2,V)

            tokens = torch.empty(B, L+2, dtype=torch.long, device=one_hot.device)
            tokens[:, 0], tokens[:, -1] = bos_id, eos_id
            tokens[:, 1:-1] = self.ids[one_hot.argmax(-1)]

            nll = -logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            return nll.mean(1) / self.T

    def _grad(self, one_hot: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        one_hot.requires_grad_()
        E = self._energy(one_hot, model)
        g = torch.autograd.grad(E.sum(), one_hot)[0]
        return g.clamp_(-self.grad_clip, self.grad_clip)

    # ------------------------------------------------ DMALA step
    def step(self, x: torch.LongTensor, model: torch.nn.Module) -> tuple[torch.LongTensor, torch.Tensor]:
        cur = x.clone()
        B = cur.size(0)
        g_inv = (1.0 / self.g).view(1, self.L, 1).expand(B, self.L, self.A)

        # Placeholder in case n_steps is 0, though it's usually >= 1
        avg_entropy_per_chain = torch.zeros(B, device=x.device)

        for _ in range(self.n_steps):
            hot   = self._one_hot(cur)
            grad  = self._grad(hot, model)
            idx   = cur.unsqueeze(-1)
            grad0 = grad.gather(2, idx)

            penalty = (g_inv / self.alpha).clone()
            penalty.scatter_(2, idx, 0)
            logits_f = -(self.alpha/2)*(grad-grad0) - penalty

            # --- Entropy Calculation ---
            dist = torch.distributions.Categorical(logits=logits_f)
            entropy_per_pos = dist.entropy() # Shape: (B, L)
            # Average entropy for the mutable part of each chain
            avg_entropy_per_chain = entropy_per_pos[:, self.prefix:].mean(dim=1) # Shape: (B,)
            
            prop = dist.sample()

            if self.prefix:  # keep prefix frozen
                 prop[:, :self.prefix] = cur[:, :self.prefix]

            # ─── hard cap on simultaneous flips ────────────────
            # in step() after prefix-freeze
            if self.k > 0:
                diff = (prop != cur)                       # (B,L) bool
                diff[:, :self.prefix] = False              # ignore frozen part
                over = diff.sum(1) > self.k
                if over.any():
                    # importance = |grad| per site, again ignore prefix
                    imp = grad.abs().amax(2)
                    imp[:, :self.prefix] = -1e9            # never choose frozen
                    topk = imp.topk(self.k, dim=1).indices
                    keep = torch.zeros_like(diff); keep.scatter_(1, topk, True)
                    revert = over.unsqueeze(-1) & diff & (~keep)
                    prop[revert] = cur[revert]

            # reverse proposal
            hot_p   = self._one_hot(prop)
            grad_p  = self._grad(hot_p, model)
            idx_p   = prop.unsqueeze(-1)
            grad_p0 = grad_p.gather(2, idx_p)
            penalty_r = (g_inv / self.alpha).clone()
            penalty_r.scatter_(2, idx_p, 0)
            logits_r = -(self.alpha/2)*(grad_p-grad_p0) - penalty_r

            q_fwd = dist # Re-use the distribution object
            q_rev = torch.distributions.Categorical(logits=logits_r)

            # Exclude frozen prefix from proposal log-prob sums to maintain detailed balance
            logprob_rev = q_rev.log_prob(cur)
            logprob_fwd = q_fwd.log_prob(prop)
            if self.prefix:
                logprob_rev = logprob_rev[:, self.prefix:]
                logprob_fwd = logprob_fwd[:, self.prefix:]

            logA = (
                self._energy(hot, model)
                - self._energy(hot_p, model)
                + logprob_rev.sum(1)
                - logprob_fwd.sum(1)
            )
            accept = (torch.rand_like(logA).log() < logA)
            self.acc_rate.mul_(self.decay).add_(accept.float().mean()*(1-self.decay))
            cur[accept] = prop[accept]

        return cur, avg_entropy_per_chain


class LangevinSearch:
    """Multi-chain driver around `DMALAProteinSampler`."""

    def __init__(
        self,
        model: EsmForMaskedLM,
        tokenizer: EsmTokenizer,
        device,
        *,
        step_size: float = 0.35,
        temp: float = 1.0,
        constant_prefix: int = 19,
        preconditioner: torch.Tensor | None = None,
        max_flips: int = 0,
    ):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.aa_tokens = [tok for tok in tokenizer.get_vocab()
                          if len(tok) == 1 and tok.isupper()]
        self.aa_ids = torch.tensor([tokenizer.convert_tokens_to_ids(t)
                                    for t in self.aa_tokens], device=device)
        self.step_size, self.temp = step_size, temp
        self.constant_prefix = constant_prefix
        self.precond = preconditioner
        self.max_flips = max_flips

    # ---------- helpers
    def encode(self, seq: str) -> torch.LongTensor:
        return torch.tensor([[self.aa_tokens.index(c) for c in seq]], device=self.device)

    def decode(self, arr: torch.LongTensor) -> str:
        return ''.join(self.aa_tokens[i] for i in arr.tolist())

    def score(self, seqs: List[str]) -> np.ndarray:
        ids = self.tokenizer(seqs, return_tensors="pt", padding=True)["input_ids"].to(self.device)
        with torch.no_grad(), accelerator.autocast():
            lp = self.model(ids).logits.log_softmax(-1)
        ll = lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        return ll[:, 1:-1].mean(1).cpu().numpy()

    # ---------- main loop
    def run(self, start_seq: str, *, steps=1000, num_chains=128, print_int=10):
        L = len(start_seq)
        sampler = DMALAProteinSampler(
            seq_len=L,
            aa_token_ids=self.aa_ids,
            step_size=self.step_size,
            temp=self.temp,
            preconditioner=self.precond,
            constant_prefix=self.constant_prefix,
            max_flips=self.max_flips,
        ).to(self.device)

        cur   = torch.tile(self.encode(start_seq), (num_chains, 1))
        best  = cur.clone()
        best_scores = self.score([start_seq]*num_chains)
        
        # --- History tracking ---
        entropy_history = torch.zeros((steps, num_chains))

        for step in tqdm(range(steps), desc="Langevin DMALA"):
            cur, entropy_per_chain = sampler.step(cur, self.model)
            entropy_history[step, :] = entropy_per_chain.cpu()
            
            scores = self.score([self.decode(r) for r in cur])
            better = scores > best_scores
            if better.any():
                best_scores[better] = scores[better]
                best[better] = cur[better]

            if (step + 1) % print_int == 0:
                acc = sampler.acc_rate.item()
                ent_str = f"| entropy {entropy_per_chain.mean().item():.3f}"
                print(f"step {step+1:4d} | best {best_scores.max():.3f} | acc {acc:.2%}" + ent_str)

        return [self.decode(r) for r in best], best_scores, entropy_history.numpy()

# ---------------------------------------------------------------------
# MASLAProteinSampler  —  sub-gradient Metropolis-Adjusted Langevin
# ---------------------------------------------------------------------
class MASLAProteinSampler(DMALAProteinSampler):
    """
    Identical interface to DMALA, but uses a *sub-gradient*:

      g̃ = { ∇U             if ||∇U||₁ > 0
          { ε·sign(rnd)    otherwise   (ε ≈ 1e-3)

    so it remains valid even if some energy terms are piece-wise constant.
    """
    def __init__(self, seq_len: int, aa_token_ids: torch.LongTensor, aa_tokens: List[str],
                 *args, penalties: list = [], penalty_mode: str = "token", **kwargs):
        super().__init__(seq_len, aa_token_ids, *args, **kwargs)
        assert penalty_mode in ("token", "predictive"), "penalty_mode must be 'token' or 'predictive'"
        self.penalty_mode = penalty_mode
        self.penalties = []
        self._aa_tokens = aa_tokens
        for pos, char, val in penalties:
            try:
                aa_idx = aa_tokens.index(char)
            except ValueError:
                raise ValueError(f"Invalid amino acid '{char}' in penalty")
            self.penalties.append( (pos - 1, aa_idx, float(val)) )


    def _energy(self, one_hot: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """
        U  =  U_PLM(one_hot) / T
            + Σ_i penalty_i

        penalty_i choices:
          - token mode:      λ_i * (1 – 1{token_i == target})
          - predictive mode: λ_i * ( - log p_model(target at i | context) )
        """
        with torch.cuda.amp.autocast(enabled=False):
            B, L, _ = one_hot.shape
            W = model.get_input_embeddings().weight                 # (V,d)
            emb = torch.einsum("bla,ad->bld", one_hot, W[self.ids]) # (B,L,d)

            bos_id = model.config.bos_token_id or 1
            eos_id = model.config.eos_token_id or 2
            emb = torch.cat([W[bos_id].expand(B,1,-1),
                             emb,
                             W[eos_id].expand(B,1,-1)], dim=1)      # (B,L+2,d)

            # add positional embeddings
            pos_emb = model.esm.embeddings.position_embeddings.weight[:L+2]
            emb = emb + pos_emb.unsqueeze(0).to(emb.dtype)

            hidden = model.esm.encoder(emb, return_dict=True).last_hidden_state
            logits = model.lm_head(hidden).log_softmax(-1)          # (B,L+2,V)

            tokens = torch.empty(B, L+2, dtype=torch.long, device=one_hot.device)
            tokens[:, 0], tokens[:, -1] = bos_id, eos_id
            tokens[:, 1:-1] = self.ids[one_hot.argmax(-1)]

            nll = -logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            E = nll.mean(1) / self.T

        if not self.penalties:
            return E

        if self.penalty_mode == "token":
            # Linear soft-penalty based on the actual token at the site
            for pos, aa_idx, lam in self.penalties:
                p_hit = one_hot[:, pos, aa_idx]  # (B,)
                E = E + lam * (1.0 - p_hit)
            return E
        else:  # predictive
            # Encourage high probability for the target AA at constrained sites
            # Use a SINGLE masked forward pass that masks all constrained positions at once
            mask_id = getattr(model.config, "mask_token_id", None)
            if mask_id is None:
                raise ValueError("predictive penalty requires model.config.mask_token_id to be set")

            with torch.cuda.amp.autocast(enabled=False):
                B2, L2, _ = one_hot.shape
                W2 = model.get_input_embeddings().weight
                # Build embeddings from current one-hot
                emb_m = torch.einsum("bla,ad->bld", one_hot, W2[self.ids])  # (B,L,d)
                # Mask all constrained positions in the same pass (BOS/EOS are added later)
                for pos, _aa_idx, _lam in self.penalties:
                    emb_m[:, pos, :] = W2[mask_id]

                # Add BOS/EOS and positional embeddings
                emb_m = torch.cat([W2[bos_id].expand(B2,1,-1), emb_m, W2[eos_id].expand(B2,1,-1)], dim=1)
                pos_emb2 = model.esm.embeddings.position_embeddings.weight[:L2+2]
                emb_m = emb_m + pos_emb2.unsqueeze(0).to(emb_m.dtype)

                # Forward encoder once and compute log-probs
                hidden_m = model.esm.encoder(emb_m, return_dict=True).last_hidden_state
                logp_m = model.lm_head(hidden_m).log_softmax(-1)  # (B, L+2, V)

            # Gather and add all predictive penalties
            total_pen = 0.0
            for pos, aa_idx, lam in self.penalties:
                vocab_id = self.ids[aa_idx]
                tlogp = logp_m[:, pos+1, vocab_id]
                total_pen = total_pen + (-lam) * tlogp
            E = E + total_pen
            return E

    def _grad(self, one_hot: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        one_hot.requires_grad_()
        E = self._energy(one_hot, model)
        g = torch.autograd.grad(E.sum(), one_hot, retain_graph=False)[0]

        # ---- sub-gradient fallback: if a whole row is zero, inject tiny noise not exactly MASLA but good enough
        mask_zero = (g.abs().sum(dim=2, keepdim=True) == 0)      # (B,L,1)
        if mask_zero.any():
            eps = 1e-3 * torch.sign(torch.randn_like(g))
            g = g + mask_zero * eps

        return g.clamp_(-self.grad_clip, self.grad_clip)


# ---------------------------------------------------------------------
# MASLASearch driver  (mirrors LangevinSearch)
# ---------------------------------------------------------------------
class MASLASearch(LangevinSearch):
    def __init__(
        self,
        model: EsmForMaskedLM,
        tokenizer: EsmTokenizer,
        device,
        *,
        step_size: float = 0.35,
        temp: float = 1.0,
        constant_prefix: int = 19,
        preconditioner: torch.Tensor | None = None,
        max_flips: int = 0,
        penalties: list = [],
        penalty_mode: str = "token",
    ):
        super().__init__(model, tokenizer, device, 
                         step_size=step_size, temp=temp, 
                         constant_prefix=constant_prefix, 
                         preconditioner=preconditioner, 
                         max_flips=max_flips)
        self.penalties = penalties
        self.penalty_mode = penalty_mode

    def run(self, start_seq: str, *, steps=1000, num_chains=128, print_int=10):
        L = len(start_seq)
        sampler = MASLAProteinSampler(
            seq_len=L,
            aa_token_ids=self.aa_ids,
            aa_tokens=self.aa_tokens,
            step_size=self.step_size,
            temp=self.temp,
            preconditioner=self.precond,
            constant_prefix=self.constant_prefix,
            max_flips=self.max_flips,
            penalties=self.penalties,
            penalty_mode=self.penalty_mode,
        ).to(self.device)

        cur   = torch.tile(self.encode(start_seq), (num_chains, 1))
        best  = cur.clone()
        best_scores = self.score([start_seq] * num_chains)

        # --- History tracking ---
        entropy_history = torch.zeros((steps, num_chains))

        for step in tqdm(range(steps), desc="MASLA"):
            cur, entropy_per_chain = sampler.step(cur, self.model)
            entropy_history[step, :] = entropy_per_chain.cpu()

            scores = self.score([self.decode(r) for r in cur])
            mask = scores > best_scores
            if mask.any():
                best_scores[mask] = scores[mask]
                best[mask] = cur[mask]
            
            if (step + 1) % print_int == 0:
                acc = sampler.acc_rate.item()
                ent_str = f"| entropy {entropy_per_chain.mean().item():.3f}"
                print(f"step {step+1:4d} | best {best_scores.max():.3f} "
                      f"| acc {acc:.2%}" + ent_str)

        return [self.decode(r) for r in best], best_scores, entropy_history.numpy()

class SequenceMCMC:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        constant_prefix=20,
        base_acc=0.90,
    ):
        self.model       = model
        self.tokenizer   = tokenizer
        self.device      = device
        self.mask_id     = tokenizer.mask_token_id
        self.const_pref  = constant_prefix
        self.base_acc    = base_acc

        # one‑letter amino‑acid tokens in ESM tokenizer
        self.aa_tokens = [
            tok for tok in tokenizer.get_vocab()
            if len(tok) == 1 and tok.isupper()
        ]

    # -------- batched masked‑marginal score --------
    @torch.no_grad()
    def score(self, seqs):
        enc = self.tokenizer(seqs, return_tensors='pt', padding=True)
        ids = enc['input_ids'].to(self.device)   # (N,L)
        N, L = ids.size()
        scores = torch.zeros(N, device=self.device)

        with torch.no_grad(), accelerator.autocast():
            for pos in range(1, L-1):            # mask one column at a time
                masked = ids.clone()
                masked[:, pos] = self.mask_id
                logits = self.model(masked).logits[:, pos, :]   # (N,V)
                logp   = torch.log_softmax(logits, dim=-1)
                true   = ids[:, pos]
                scores += logp[torch.arange(N), true]

        return (scores / (L-2)).cpu().numpy()

    # -------- random mutator --------
    def mutate(self, seq, max_mut=2):
        if len(seq) <= self.const_pref:
            raise ValueError("Sequence shorter than constant prefix")
        mutable = list(range(self.const_pref, len(seq)))
        n = random.randint(1, min(max_mut, len(mutable)))
        pos = random.sample(mutable, n)
        seq_list = list(seq)
        for p in pos:
            choices = [aa for aa in self.aa_tokens if aa != seq_list[p]]
            seq_list[p] = random.choice(choices)
        return ''.join(seq_list)

    # -------- simulated‑annealing loop --------
    def run(
        self,
        start_seq,
        steps=1000,
        beta_start=2.0,
        beta_end=50.0,
        max_mutations=2,
        num_chains=128,
        print_int=10,
    ):
        chains  = [start_seq] * num_chains
        scores  = self.score(chains)
        best_s  = scores.copy()
        best_q  = chains.copy()
        accept_tot = 0

        betas = beta_start * (beta_end / beta_start) ** (
                np.arange(1, steps + 1) / steps)

        for step, beta in enumerate(tqdm(betas, desc="Simulated annealing"), 1):
            props = [self.mutate(s, max_mutations) for s in chains]
            prop_scores = self.score(props)
            deltas = prop_scores - scores

            # calibrated acceptance
            sigma  = stable_sigmoid(beta * deltas)
            probs  = self.base_acc * sigma + (1 - self.base_acc) * (1 - sigma)
            accept = np.random.rand(num_chains) < probs

            for i, ok in enumerate(accept):
                if ok:
                    chains[i], scores[i] = props[i], prop_scores[i]
                    accept_tot += 1
                    if scores[i] > best_s[i]:
                        best_s[i], best_q[i] = scores[i], chains[i]

            if step % print_int == 0:
                acc_rate = accept_tot / (step * num_chains)
                print(f"step {step:4d} | β {beta:6.2f} | "
                      f"best {best_s.max():7.3f} | acc {acc_rate:5.2%}")

        return best_q, best_s

# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", default="facebook/esm1v_t33_650M_UR90S_1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wt_seq", default="MKKFRWVVLVVVVLACLLLWAQVFNMMCDQDVQFFSGICAINQFIPW")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num_chains", type=int, default=128)
    parser.add_argument("--sampler",
                    choices=["langevin", "masla", "sa"],
                    default="langevin",
                     help="langevin = DMALA, masla = sub-gradient MALA, sa = simulated annealing")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--fasta_out", default="best_variants.fasta")
    parser.add_argument("--alpha",  type=float, default=0.25, help="DMALA step size")
    parser.add_argument("--temp",   type=float, default=1.0,  help="Temperature T")
    parser.add_argument("--k_flips",type=int,   default=0,    help="Max flips per sweep (0 = no cap)")
    parser.add_argument("--prefix", type=int, default=20, help="Constant prefix")
    parser.add_argument("--penalties", type=str, default="",
                        help="MASLA only: constraints e.g. '28,C,1000;39,C,1000'")
    parser.add_argument("--penalty_mode", type=str, choices=["token", "predictive"], default="token",
                        help="MASLA only: 'token' uses 1-hit soft penalty; 'predictive' uses -log p_model(target) at site")
    args = parser.parse_args()

    global accelerator
    accelerator = Accelerator()
    device = accelerator.device

    print("Loading tokenizer …")
    tokenizer = EsmTokenizer.from_pretrained(args.model_name, do_lower_case=False)
    print("Loading base model …")
    base_model = EsmForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True)
    print(f"Applying LoRA weights from '{args.checkpoint}' …")
    model = PeftModel.from_pretrained(base_model, args.checkpoint).eval()
    model = accelerator.prepare(model)

    wt_score = None
    with torch.no_grad(), accelerator.autocast():
        wid = tokenizer([args.wt_seq], return_tensors="pt")["input_ids"].to(device)
        wt_lp = model(wid).logits.log_softmax(-1)
        wt_ll = wt_lp.gather(-1, wid.unsqueeze(-1)).squeeze(-1)
        wt_score = wt_ll[:, 1:-1].mean().item()
    print(f"WT average log-prob: {wt_score:.3f}")

    history = None # Will hold per-chain entropy history
    if args.sampler in ["langevin", "masla"]:
        if args.sampler == "langevin":
            search = LangevinSearch(
                model, tokenizer, device,
                step_size=args.alpha,
                temp=args.temp,
                max_flips=args.k_flips,
                constant_prefix=args.prefix,
            )
        else: # masla
            penalties = []
            if args.penalties:
                try:
                    for p in args.penalties.split(';'):
                        pos, char, val = p.split(',')
                        penalties.append( (int(pos), char, float(val)) )
                except ValueError:
                    raise ValueError(f"Could not parse penalty string: '{args.penalties}'")
            search = MASLASearch(
                model, tokenizer, device,
                step_size=args.alpha,
                temp=args.temp,
                max_flips=args.k_flips,
                constant_prefix=args.prefix,
                penalties=penalties,
                penalty_mode=args.penalty_mode,
            )
        best_seqs, best_scores, history = search.run(
            args.wt_seq,
            steps=args.steps,
            num_chains=args.num_chains,
            print_int=10,
        )
    else: # args.sampler == "sa"
        sa = SequenceMCMC(model, tokenizer, device)
        best_seqs, best_scores = sa.run(
            args.wt_seq,
            steps=args.steps,
            num_chains=args.num_chains,
        )

    # --------------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------------
    top_idx = np.argsort(best_scores)[-args.top_n:][::-1]
    with open(args.fasta_out, "w") as fh:
        fh.write(f">variant_000|WT|score={wt_score:.4f}\n")
        fh.write("\n".join(textwrap.wrap(args.wt_seq, 60)) + "\n")
        for rank, idx in enumerate(top_idx, 1):
            fh.write(f">variant_{rank:03d}|score={best_scores[idx]:.4f}\n")
            fh.write("\n".join(textwrap.wrap(best_seqs[idx], 60)) + "\n")
    print(f"Wrote top {args.top_n} variants to '{args.fasta_out}'")

    if history is not None:
        plt.figure(figsize=(12, 7))
        
        num_chains = history.shape[1]
        top_n_chains = 50
        top_n_chains = min(top_n_chains, num_chains)
        top_indices = np.argsort(best_scores)[-top_n_chains:]

        # Plot all chains' entropy histories in gray
        plt.plot(history, color='gray', alpha=0.2, linewidth=0.5)
        # Highlight top chains in red
        if top_n_chains > 0:
            plt.plot(history[:, top_indices], color='red', alpha=0.3, linewidth=0.5)
        
        # Plot the mean of all chains
        plt.plot(history.mean(axis=1), color='black', linestyle='--', label=f'Mean ({num_chains} chains)')
        # Plot the mean of the top chains
        if top_n_chains > 0:
            plt.plot(history[:, top_indices].mean(axis=1), color='red', linestyle='--', label=f'Mean (Top {top_n_chains} chains)')

        plt.title(f"Per-Chain Proposal Entropy Convergence ({args.sampler.upper()})")
        plt.xlabel("MCMC Step")
        plt.ylabel("Avg. Proposal Entropy (per chain)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plot_filename = os.path.splitext(args.fasta_out)[0] + "_chain_entropy.png"
        plt.savefig(plot_filename)
        print(f"Saved chain entropy plot to '{plot_filename}'")


if __name__ == "__main__":
    main()
