#!/usr/bin/env python3
"""
Plot spectra from one or more text files containing two columns: k  E(k).

Usage:
  python plot_txt_spectra.py file1.txt file2.txt --save spectrum_from_txt.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_spectrum(path):
    """Load spectrum with two columns k, E(k); handles whitespace or CSV; skips # comments."""
    try:
        data = np.loadtxt(path, comments="#", delimiter=None)
    except Exception:
        # fallback: try CSV
        data = np.loadtxt(path, comments="#", delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] >= 2:
        k = data[:, 0]
        ek = data[:, 1]
    else:
        raise ValueError("Expected at least 2 columns for k and E(k)")
    return k, ek


def main():
    ap = argparse.ArgumentParser(description="Plot spectra from text files (k E(k) columns).")
    ap.add_argument("files", nargs="+", help="Text files with two columns: k  E(k)")
    ap.add_argument("--save", type=str, default=None, help="Path to save the plot (PNG).")
    ap.add_argument("--title", type=str, default="Spectra", help="Plot title.")
    args = ap.parse_args()

    plt.figure(figsize=(7, 5))
    for path in args.files:
        if not os.path.isfile(path):
            print(f"[WARN] Skipping missing file: {path}")
            continue
        try:
            k, ek = load_spectrum(path)
        except Exception as exc:
            print(f"[WARN] Failed to read {path}: {exc}")
            continue
        mask = (k > 0) & (ek > 0)
        if not np.any(mask):
            print(f"[WARN] No positive data in {path}")
            continue
        plt.loglog(k[mask], ek[mask], label=os.path.basename(path))

    plt.xlabel("k")
    plt.ylabel("E(k)")
    plt.title(args.title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
