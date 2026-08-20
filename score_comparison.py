import argparse
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

from transformers import EsmForMaskedLM, EsmTokenizer
from peft import PeftModel
from accelerate import Accelerator

warnings.filterwarnings('ignore', message='The argument `trust_remote_code` is to be used with Auto classes.*')

def parse_args():
    parser = argparse.ArgumentParser(description="Compare MCMC scoring with masked pseudo-likelihood.")
    parser.add_argument("--model_name", type=str, default="facebook/esm1v_t33_650M_UR90S_1", help="Base ESM model.")
    parser.add_argument("--datasetname", type=str, required=True, help="Dataset name to find lora checkpoint and data.")
    parser.add_argument("--lora_seed", type=int, default=1, help="Seed for the LoRA checkpoint.")
    parser.add_argument("--batch_size_pl", type=int, default=32, help="Batch size for Pseudo-Likelihood scoring.")
    parser.add_argument("--batch_size_mpl", type=int, default=8, help="Batch size for Masked Pseudo-Likelihood scoring.")
    parser.add_argument("--output_plot", type=str, default="scoring_correlation.png", help="Path to save the output plot.")
    return parser.parse_args()

@torch.no_grad()
def score_pseudo_likelihood(model, tokenizer, seqs, accelerator, batch_size=32):
    """Computes avg. log-likelihood over non-special tokens."""
    device = accelerator.device
    all_scores = []
    for i in tqdm(range(0, len(seqs), batch_size), desc="Scoring Pseudo-Likelihood"):
        batch_seqs = seqs[i:i+batch_size]
        ids = tokenizer(batch_seqs, return_tensors="pt", padding=True)["input_ids"].to(device)
        with accelerator.autocast():
            logits = model(ids).logits
            lp = logits.log_softmax(-1)
        ll = lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        scores = ll[:, 1:-1].mean(1).cpu().numpy()
        all_scores.extend(scores)
    return np.array(all_scores)

@torch.no_grad()
def score_masked_pseudo_likelihood(model, tokenizer, seqs, accelerator, batch_size=8):
    """Computes masked pseudo-log-likelihood (avg. over positions)."""
    device = accelerator.device
    mask_id = tokenizer.mask_token_id
    all_scores = []

    for i in tqdm(range(0, len(seqs), batch_size), desc="Scoring Masked PL"):
        batch_seqs = seqs[i:i+batch_size]
        try:
            enc = tokenizer(batch_seqs, return_tensors='pt', padding=True)
        except Exception as e:
            print(f"Tokenizer error on batch starting with: {batch_seqs[0]}")
            continue

        ids = enc['input_ids'].to(device)
        attn_mask = enc['attention_mask'].to(device)
        N, L = ids.shape
        batch_scores = torch.zeros(N, device=device)
        
        if L <= 2: # only BOS/EOS
            all_scores.extend([np.nan] * N)
            continue

        with accelerator.autocast():
            for pos_to_mask in range(1, L - 1):
                masked_ids = ids.clone()
                masked_ids[:, pos_to_mask] = mask_id
                logits_at_pos = model(input_ids=masked_ids, attention_mask=attn_mask).logits[:, pos_to_mask, :]
                logp = torch.log_softmax(logits_at_pos, dim=-1)
                true_tokens_at_pos = ids[:, pos_to_mask]
                batch_scores += logp.gather(-1, true_tokens_at_pos.unsqueeze(-1)).squeeze(-1)
        
        avg_scores = (batch_scores / (L - 2)).cpu().numpy()
        all_scores.extend(avg_scores)

    return np.array(all_scores)

def main():
    args = parse_args()

    print("--- Initializing Accelerator, Model, and Tokenizer ---")
    accelerator = Accelerator()
    
    LORA_CHECKPOINT = f"checkpoint/{args.datasetname}/seed{args.lora_seed}"

    tokenizer = EsmTokenizer.from_pretrained(args.model_name, do_lower_case=False)
    base_model = EsmForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT).eval()
    model = accelerator.prepare(model)
    print(f"Loaded model with LoRA weights from: {LORA_CHECKPOINT}")

    print("\n--- Loading Data ---")
    test_csv_path = f"data/{args.datasetname}/test.csv"
    try:
        test_csv = pd.read_csv(test_csv_path)
    except FileNotFoundError:
        print(f"Error: Test data not found at {test_csv_path}")
        return
        
    sequences = test_csv['sequence'].unique().tolist()
    print(f"Found {len(sequences)} unique sequences in the test set.")
    
    sequences = [s for s in sequences if isinstance(s, str) and len(s) > 0]
    print(f"Processing {len(sequences)} valid sequences.")

    print("\n--- Calculating Scores ---")
    pl_scores = score_pseudo_likelihood(model, tokenizer, sequences, accelerator, args.batch_size_pl)
    mpl_scores = score_masked_pseudo_likelihood(model, tokenizer, sequences, accelerator, args.batch_size_mpl)

    valid_indices = ~np.isnan(pl_scores) & ~np.isnan(mpl_scores)
    pl_scores_valid = pl_scores[valid_indices]
    mpl_scores_valid = mpl_scores[valid_indices]

    print("\n--- Correlation and Plotting ---")
    if len(pl_scores_valid) < 2:
        print("Not enough valid data points to calculate correlation.")
        return
        
    corr, p_val = spearmanr(pl_scores_valid, mpl_scores_valid)

    plt.figure(figsize=(8, 6))
    plt.scatter(pl_scores_valid, mpl_scores_valid, alpha=0.5, edgecolor='k', s=40)
    plt.title(f"Scoring Method Correlation on Test Set ('{args.datasetname}')")
    plt.xlabel("Pseudo-Likelihood (Langevin Sampler Style)")
    plt.ylabel("Masked Pseudo-Likelihood (Training Style)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.text(0.05, 0.95, f'Spearman ρ = {corr:.3f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    plt.savefig(args.output_plot)
    print(f"Plot saved to {args.output_plot}")

    print(f"\nSpearman Correlation: {corr:.4f}")
    print(f"P-value: {p_val:.4g}")

if __name__ == "__main__":
    main()
