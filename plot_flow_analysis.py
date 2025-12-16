#!/usr/bin/env python3
"""
Minimal utility: load an ETA HDF5 and print kinetic energy stats.
"""
import argparse
import glob
import os
import re
import sys

# Avoid OpenMP shared-memory issues in restricted sandboxes (and keep plotting lightweight).
os.environ.setdefault("KMP_SHM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import h5py
import numpy as np
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
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


def print_moments(infile: str, fields=("u", "KinEnDensity")):
    with h5py.File(infile, "r") as f:
        def fmt(x):
            return f"{x:.6e}" if x is not None else "n/a"

        for name in fields:
            grp = f.get(f"{name}/moments")
            if grp is None:
                continue

            def g(key):
                return float(np.array(grp[key]).squeeze()) if key in grp else None

            mean = g("mean")
            rms = g("rms")
            std = g("stddev")
            var = g("var")
            skew = g("skew")
            kurt = g("kurt")
            mn = g("min")
            mx = g("max")
            absmin = g("absmin")
            absmax = g("absmax")

            print(
                f"[{name} moments] mean={fmt(mean)} rms={fmt(rms)} std={fmt(std)} "
                f"var={fmt(var)} skew={fmt(skew)} kurt={fmt(kurt)} "
                f"min={fmt(mn)} max={fmt(mx)} absmin={fmt(absmin)} absmax={fmt(absmax)}"
            )


def list_moment_fields(infile: str):
    fields = []
    with h5py.File(infile, "r") as f:
        for name, grp in f.items():
            if isinstance(grp, h5py.Group) and "moments" in grp:
                fields.append(name)
    return sorted(fields)


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

def collect_histograms(infile: str):
    """
    Collect histograms written by FlowAnalysis (hist/* datasets).
    Returns list of (name, bins, counts).
    """
    hists = []
    with h5py.File(infile, "r") as f:
        for name, grp in f.items():
            if not isinstance(grp, h5py.Group):
                continue
            if "hist" not in grp:
                continue
            for hname, dset in grp["hist"].items():
                arr = np.array(dset)
                if arr.shape[0] != 2:
                    continue
                bins = arr[0]
                counts = arr[1][:len(bins)-1]  # ignore last padding slot
                hists.append((f"{name}/hist/{hname}", bins, counts))
    return hists

def plot_histograms(hists, show=False, save=False):
    for name, bins, counts in hists:
        if counts.sum() == 0:
            continue
        pdf = counts / counts.sum() / np.diff(bins)
        plt.figure(figsize=(6,4))
        plt.step(bins[:-1], counts, where="post", label="hist")
        plt.step(bins[:-1], pdf, where="post", label="pdf (norm)")
        plt.xlabel(name)
        plt.ylabel("count / pdf")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        if save:
            safe_name = name.replace("/", "_")
            out = f"{safe_name}.png"
            plt.savefig(out, dpi=150)
            print(f"Wrote {out}")
        if show:
            plt.show()
        plt.close()

def _list_1d_hist_names(f: h5py.File, field: str):
    grp = f.get(f"{field}/hist")
    if grp is None:
        return []
    names = []
    for hname, obj in grp.items():
        if not isinstance(obj, h5py.Dataset):
            continue
        arr = np.array(obj)
        if arr.ndim != 2 or arr.shape[0] != 2:
            continue
        names.append(hname)
    return names


def _resolve_hist_name(available: list, preferred: str):
    if not available:
        return None
    if preferred in (None, "", "auto"):
        for cand in ("globalMinMaxMinMax", "SimMinMax", "SnapMinMax"):
            if cand in available:
                return cand
        return sorted(available)[0]

    base_map = {
        "globalMinMax": "globalMinMaxMinMax",
        "Snap": "SnapMinMax",
        "Sim": "SimMinMax",
    }
    target = base_map.get(preferred, preferred)
    if target not in available and (target + "MinMax") in available:
        target = target + "MinMax"
    return target if target in available else None


def load_1d_hist(infile: str, field: str, histbins: str = "auto"):
    with h5py.File(infile, "r") as f:
        available = _list_1d_hist_names(f, field)
        hname = _resolve_hist_name(available, histbins)
        if hname is None:
            return None
        arr = np.array(f[f"{field}/hist/{hname}"])
        bins = arr[0]
        vals = arr[1]
        return hname, bins, vals


def _hist_to_pdf(bins: np.ndarray, vals: np.ndarray):
    n = min(len(vals), len(bins) - 1)
    widths = np.diff(bins[: n + 1])
    total = float(np.sum(widths * vals[:n]))
    if not np.isfinite(total) or total <= 0:
        return None
    pdf = np.zeros_like(vals, dtype=float)
    pdf[:n] = vals[:n] / total
    return pdf


def _maybe_filter_by_time(files, dump_dt: float, min_time: float, dump_regex: str):
    if min_time <= 0 or dump_dt is None:
        return files

    parsed_any = False
    kept = []
    for fn in files:
        m = re.search(dump_regex, os.path.basename(fn))
        if not m:
            kept.append(fn)
            continue
        parsed_any = True
        dump_idx = int(m.group(1))
        if dump_idx * dump_dt < min_time:
            continue
        kept.append(fn)

    if parsed_any and not kept:
        print("[WARN] All files filtered out by --pdf-min-time; keeping original list.")
        return files
    return kept


def plot_field_pdf(
    files,
    field: str = "u",
    histbins: str = "auto",
    mode: str = "instant",
    out_png: str = None,
    show: bool = False,
    save: bool = False,
    xlim=None,
    logy: bool = True,
    dump_dt: float = None,
    min_time: float = 0.0,
    dump_regex: str = r"(\d+)(?=\.hdf5$)",
):
    files = list(files)
    files = _maybe_filter_by_time(files, dump_dt=dump_dt, min_time=min_time, dump_regex=dump_regex)
    if not files:
        return

    curves = []
    labels = []
    bins_ref = None
    hist_name_used = None

    for fn in files:
        loaded = load_1d_hist(fn, field=field, histbins=histbins)
        if loaded is None:
            continue
        hname, bins, vals = loaded
        pdf = _hist_to_pdf(bins, vals)
        if pdf is None:
            continue

        if bins_ref is None:
            bins_ref = bins
            hist_name_used = hname
        elif len(bins) != len(bins_ref) or not np.allclose(bins, bins_ref):
            print(f"[WARN] Skipping {fn}: histogram bins differ from first file.")
            continue

        curves.append(pdf)
        labels.append(os.path.basename(fn))

    if not curves or bins_ref is None:
        print(f"[WARN] No usable 1D histograms found for field '{field}'.")
        return

    x = bins_ref[:-1]

    plt.figure(figsize=(6, 3.8))
    if mode == "mean":
        stack = np.stack(curves, axis=0)
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)
        plt.plot(x, mean[:-1], label=f"{field} (mean, {len(curves)} files)")
        if len(curves) > 1:
            plt.fill_between(x, (mean - std)[:-1], (mean + std)[:-1], alpha=0.3)
    else:
        for pdf, lab in zip(curves, labels):
            plt.plot(x, pdf[:-1], label=lab)

    plt.ylabel("PDF")
    if field == "u":
        plt.xlabel(r"velocity $u$")
    else:
        plt.xlabel(field)
    plt.grid(True, alpha=0.3)
    if logy:
        plt.yscale("log")
    if xlim is not None:
        plt.xlim(float(xlim[0]), float(xlim[1]))
    if mode != "mean" or len(curves) > 1:
        plt.legend(loc="best", fontsize=8)
    plt.title(f"{field} histogram ({hist_name_used})")
    plt.tight_layout()
    if save:
        out_png = out_png or f"{field}_pdf.png"
        plt.savefig(out_png, dpi=150)
        print(f"Wrote {out_png}")
    if show:
        plt.show()
    plt.close()

def plot_split_spectrum(spectra: dict, out_png: str, show: bool, save: bool):
    """Plot compressive and rotational spectra separately (no total curve)."""
    if spectra.get("dil") is None and spectra.get("sol") is None:
        return
    k = spectra["k"]
    plt.figure(figsize=(7, 4))
    if spectra.get("dil") is not None:
        mask = spectra["dil"] > 0
        plt.loglog(k[mask], spectra["dil"][mask] + 1e-300, "--", label="Compressive (Dil)")
    if spectra.get("sol") is not None:
        mask = spectra["sol"] > 0
        plt.loglog(k[mask], spectra["sol"][mask] + 1e-300, ":", label="Rotational (Sol)")
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

def plot_compensated_spectrum(spectra: dict, out_png: str, show: bool, save: bool):
    """Plot k^(5/3) * E(k) for total, compressive, rotational."""
    k = spectra["k"]
    comp = k ** (5.0/3.0)
    plt.figure(figsize=(7, 4))
    if spectra.get("total") is not None:
        mask = spectra["total"] > 0
        plt.loglog(k[mask], spectra["total"][mask]*comp[mask] + 1e-300, label="Total (compensated)")
    if spectra.get("dil") is not None:
        mask = spectra["dil"] > 0
        plt.loglog(k[mask], spectra["dil"][mask]*comp[mask] + 1e-300, "--", label="Compressive (compensated)")
    if spectra.get("sol") is not None:
        mask = spectra["sol"] > 0
        plt.loglog(k[mask], spectra["sol"][mask]*comp[mask] + 1e-300, ":", label="Rotational (compensated)")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$k^{5/3} E(k)$")
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
    ap.add_argument("--file", type=str, nargs="+", default=None, help="HDF5 file(s) to read (default: last matching pattern).")
    ap.add_argument("--pattern", type=str, default="flow_output*.hdf5", help="Glob pattern if --file not given.")
    ap.add_argument("--save-plot", action="store_true", help="Save spectrum plot to spectrum.png")
    ap.add_argument("--show-plot", action="store_true", help="Show spectrum plot interactively")
    ap.add_argument("--txt", type=str, default="spectrum.txt", help="Write k,E(k) to this text file")
    ap.add_argument("--plot-hist", action="store_true", help="Plot histograms/PDFs for available fields")
    ap.add_argument("--save-hist", action="store_true", help="Save histogram plots to PNGs")
    ap.add_argument(
        "--moments-fields",
        type=str,
        nargs="+",
        default=["u", "KinEnDensity", "KinEnSpecific", "AbsDivU", "AbsRotU"],
        help="Fields to print 1-point moments for (use: all).",
    )
    ap.add_argument("--no-u-pdf", action="store_true", help="Do not plot the 1D PDF for velocity magnitude u.")
    ap.add_argument("--pdf-mode", type=str, default="instant", choices=["instant", "mean"], help="How to plot PDFs across multiple files.")
    ap.add_argument("--pdf-field", type=str, default="u", help="Field name to use for the 1D PDF (default: u).")
    ap.add_argument("--pdf-histbins", type=str, default="auto", help="Histogram bin set: auto|Snap|Sim|globalMinMax or explicit dataset name.")
    ap.add_argument("--pdf-xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"), help="x-axis limits for the PDF plot.")
    ap.add_argument("--pdf-min-time", type=float, default=0.0, help="If >0 and --time-between-dumps is set, skip files with dump_index*dt < this.")
    ap.add_argument("--time-between-dumps", type=float, default=None, help="Time between dumps (used with --pdf-min-time when plotting many files).")
    ap.add_argument("--dump-regex", type=str, default=r"(\d+)(?=\.hdf5$)", help="Regex to extract dump index from filename for --pdf-min-time.")
    args = ap.parse_args()

    if args.file:
        files = args.file
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
            plot_split_spectrum(spectra, out_png="spectrum_components.png", show=args.show_plot, save=args.save_plot)
            plot_compensated_spectrum(spectra, out_png="spectrum_compensated.png", show=args.show_plot, save=args.save_plot)
        mf = args.moments_fields
        if len(mf) == 1 and mf[0].lower() == "all":
            mf = list_moment_fields(fn)
        print_moments(fn, fields=mf)
        if args.plot_hist or args.save_hist:
            hists = collect_histograms(fn)
            if not hists:
                print("[WARN] No histograms found in file.")
            else:
                plot_histograms(hists, show=args.plot_hist, save=args.save_hist)

    if (args.save_plot or args.show_plot) and not args.no_u_pdf:
        plot_field_pdf(
            files=files,
            field=args.pdf_field,
            histbins=args.pdf_histbins,
            mode=args.pdf_mode,
            out_png=None if not args.save_plot else f"{args.pdf_field}_pdf.png",
            show=args.show_plot,
            save=args.save_plot,
            xlim=args.pdf_xlim if args.pdf_xlim is not None else ((0.0, 1.3) if args.pdf_field == "u" else None),
            dump_dt=args.time_between_dumps,
            min_time=args.pdf_min_time,
            dump_regex=args.dump_regex,
        )


if __name__ == "__main__":
    main()
