#!/usr/bin/env python3
"""
Robust GReaT runner (GPU-preferred).
- Trains longer by default (epochs=60, override with EPOCHS).
- Uses 'gpt2' instead of 'distilgpt2' for more stable sampling.
- Samples shorter sequences first (max_length=600, 512).
- Uses small batch_size (5) when sampling to avoid loop breaks.
"""

import os, sys, traceback, time
import numpy as np
import pandas as pd
import torch

# ---------------- GPU info ----------------
def gpu_info():
    info = {
        "cuda": torch.cuda.is_available(),
        "n": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "name": None,
    }
    if info["cuda"] and info["n"] > 0:
        idx = torch.cuda.current_device()
        info["name"] = torch.cuda.get_device_name(idx)
    return info

cap = gpu_info()

# ---------------- import ----------------
try:
    import be_great
    from be_great import GReaT
    print("be_great imported from:", getattr(be_great, "__file__", "<unknown>"))
except Exception:
    print("Failed to import be_great. Install with: pip install -U be-great")
    raise

# ---------------- helpers ----------------
def retry_sample(model, n_total: int, maxlens, batch_size: int = 5):
    """Sample in small batches, retrying with shorter max_length if needed."""
    rows = []
    remaining = n_total

    for ml in maxlens:
        while remaining > 0:
            n = min(batch_size, remaining)
            try:
                print(f"[sample] n={n} max_length={ml}")
                out = model.sample(n_samples=n, max_length=ml)
                if out is None or len(out) == 0:
                    raise RuntimeError("Empty sample output")
                rows.append(out)
                remaining -= n
            except Exception as e:
                print(f"  -> failed at max_length={ml}: {e}")
                traceback.print_exc()
                break  # try next shorter max_length
        if remaining <= 0:
            break

    if remaining > 0:
        raise RuntimeError(f"Sampling failed; {remaining} rows left unsampled.")
    return pd.concat(rows, ignore_index=True) if isinstance(rows[0], pd.DataFrame) else pd.DataFrame(rows)

# ---------------- main ----------------
def main():
    csv_path = os.environ.get("CSV_PATH", "amg.csv")
    out_csv = os.environ.get("OUT_CSV", "synthetic.csv")
    n_samples = int(os.environ.get("N_SAMPLES", "10000"))
    epochs = int(os.environ.get("EPOCHS", "2000"))

    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(0)

    # Trim very long text cells to reduce risk of exceeding context
    max_chars_per_text_cell = int(os.environ.get("TRIM_TEXT_CHARS", "800"))
    txt_cols = df.select_dtypes(include=["object"]).columns
    for c in txt_cols:
        df[c] = df[c].astype(str).str.slice(0, max_chars_per_text_cell)

    print("CUDA available:", cap["cuda"], "| Device:", cap["name"])
    print("\nOriginal data sample:")
    print(df.head())

    # ---- train ----
    model = GReaT(
        llm="gpt2",  # stronger than distilgpt2
        batch_size=16 if cap["cuda"] else 4,
        epochs=epochs,
        dataloader_num_workers=2 if cap["cuda"] else 0,
        report_to=[],
        fp16=cap["cuda"],   # safe on GPU
    )
    model.fit(df)

    # ---- sampling ----
    # Safer lengths, start shorter to avoid loop breaks
    maxlens = [600, 512]
    print(f"\nSampling candidates: {maxlens}")

    synthetic = retry_sample(model, n_total=n_samples, maxlens=maxlens, batch_size=5)

    synthetic = synthetic.reindex(columns=df.columns, fill_value=0)
    print("\nSynthetic data preview:")
    print(synthetic.head())

    synthetic.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

if __name__ == "__main__":
    if os.path.basename(__file__) == "be_great.py":
        print("ERROR: rename this file (run_great.py).")
        sys.exit(1)
    main()

