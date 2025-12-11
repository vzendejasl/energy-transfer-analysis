#!/usr/bin/env python3
"""
Minimal utility: load an ETA HDF5 and print kinetic energy stats.
"""
import argparse
import glob
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt


def safe_get(f: h5py.File, path: str):
    if path in f:
        return f[path][()]
    alt = path[1:] if path.startswith("/") else path
    parts = [p for p in alt.split("/") if p]
    cur = f
    for p in parts:
        if p in cur:
            cur = cur[p]
        else:
            return None
    return cur[()] if isinstance(cur, h5py.Dataset) else None


def print_ke(infile: str):
    with h5py.File(infile, "r") as f:
        ke_real = safe_get(f, "u/PowSpec/TotKE_real")
        ke_spec = safe_get(f, "u/PowSpec/TotKE_spec")
        tot_full = safe_get(f, "u/PowSpec/TotFull")
        ke_mean = safe_get(f, "KinEnDensity/moments/mean")

        def to_float(x):
            return float(np.array(x).squeeze()) if x is not None else None

        kr = to_float(ke_real)
        ks = to_float(ke_spec)
        tf = to_float(tot_full)
        km = to_float(ke_mean)

        print(f"File: {infile}")
        if kr is not None and ks is not None:
            ratio = ks / kr if kr != 0 else np.nan
            print(f"  KE_real = {kr:.6e}")
            print(f"  KE_spec = {ks:.6e}")
            print(f"  ratio   = {ratio:.3f}")
        else:
            print("  KE_real/spec not found.")

        if tf is not None:
            print(f"  TotFull (sum |U|^2) = {tf:.6e}")
        if km is not None:
            print(f"  KinEnDensity mean   = {km:.6e}")


def load_vector_spectra(infile: str):
    """
    Return k centers and kinetic energy per bin (0.5*|U|^2) for total,
    compressive (Dil) and solenoidal (Sol) parts if present.
    """
    with h5py.File(infile, "r") as f:
        if "u/PowSpec/Bins" not in f or "u/PowSpec/Full" not in f:
            return None
        bins = np.array(f["u/PowSpec/Bins"])[0]
        centers = 0.5 * (bins[:-1] + bins[1:])

        def extract(path):
            if path not in f:
                return None
            arr = np.array(f[path])  # [centeredK, shell, vol, no_norm]
            return 0.5 * arr[3]  # KE per bin

        total = extract("u/PowSpec/Full")
        dil = extract("u/PowSpec/Dil")
        sol = extract("u/PowSpec/Sol")
        return {"k": centers, "total": total, "dil": dil, "sol": sol}


def write_spectrum_txt(path: str, spectra: dict):
    k = spectra["k"]
    total = spectra.get("total")
    dil = spectra.get("dil")
    sol = spectra.get("sol")
    mask = (total > 0) if total is not None else np.ones_like(k, dtype=bool)
    with open(path, "w") as fh:
        header = "# k  E_total"
        if dil is not None:
            header += "  E_dil"
        if sol is not None:
            header += "  E_sol"
        fh.write(header + "\n")
        for idx, kk in enumerate(k):
            if not mask[idx]:
                continue
            line = f"{kk:.8e}"
            if total is not None:
                line += f" {total[idx]:.8e}"
            if dil is not None:
                line += f" {dil[idx]:.8e}"
            if sol is not None:
                line += f" {sol[idx]:.8e}"
            fh.write(line + "\n")
    print(f"Wrote {path}")


def plot_ke_spectrum(spectra: dict, out_png: str, show: bool, save: bool):
    k = spectra["k"]
    plt.figure(figsize=(7, 5))
    if spectra.get("total") is not None:
        mask = spectra["total"] > 0
        plt.loglog(k[mask], spectra["total"][mask] + 1e-300, label="Total")
    if spectra.get("dil") is not None:
        mask = spectra["dil"] > 0
        plt.loglog(k[mask], spectra["dil"][mask] + 1e-300, "--", label="Compressive")
    if spectra.get("sol") is not None:
        mask = spectra["sol"] > 0
        plt.loglog(k[mask], spectra["sol"][mask] + 1e-300, ":", label="Rotational")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E(k)$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(out_png, dpi=150)
        print(f"Wrote {out_png}")
    if show:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Print kinetic energy stats from flow_output HDF5.")
    ap.add_argument("--file", type=str, default=None, help="HDF5 file to read (default: last matching pattern).")
    ap.add_argument("--pattern", type=str, default="flow_output*.hdf5", help="Glob pattern if --file not given.")
    ap.add_argument("--save-plot", action="store_true", help="Save spectrum plot to spectrum.png")
    ap.add_argument("--show-plot", action="store_true", help="Show spectrum plot interactively")
    ap.add_argument("--txt", type=str, default="spectrum.txt", help="Write k,E(k) to this text file")
    args = ap.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(args.pattern))
        if not files:
            print(f"No files match pattern: {args.pattern}")
            sys.exit(1)
        files = [files[-1]]

    for fn in files:
        print_ke(fn)
        spectra = load_vector_spectra(fn)
        if spectra is None:
            print("[WARN] Spectrum data not found in file.")
            continue
        if args.txt:
            write_spectrum_txt(args.txt, spectra)
        if args.save_plot or args.show_plot:
            plot_ke_spectrum(spectra, out_png="spectrum.png", show=args.show_plot, save=args.save_plot)


if __name__ == "__main__":
    main()
