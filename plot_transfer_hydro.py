#!/usr/bin/env python3
"""
Plot hydro energy transfer diagnostics from `transfer_output.pkl`.

Outputs:
- Shell-to-shell transfer heatmaps T(Q,K) for requested terms (e.g. UU/UUA/UUC)
- Cross-scale flux Pi(k) computed from shell-to-shell transfers (as in plots.ipynb)

Example:
  python plot_transfer_hydro.py transfer_output.pkl --show-plot
  python plot_transfer_hydro.py transfer_output.pkl --terms UU UUA UUC --save --output hydro_transfer.png
"""

import argparse
import os
import pickle
import sys

# Avoid OpenMP shared-memory issues in restricted sandboxes.
os.environ.setdefault("KMP_SHM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import matplotlib

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import logging

# Matplotlib may emit benign "posx and posy should be finite values" warnings on some layouts.
logging.getLogger("matplotlib.text").setLevel(logging.ERROR)


def load_transfer_data(pickle_file: str):
    try:
        with open(pickle_file, "rb") as f:
            try:
                return pickle.load(f, encoding="latin1")
            except TypeError:
                return pickle.load(f)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: File not found: {pickle_file}")
    except Exception as e:
        raise SystemExit(f"ERROR loading pickle file: {e}")


def _sorted_bin_strings(bin_strings):
    return sorted(bin_strings, key=lambda s: float(s.split("-")[0]))


def _parse_edges(bin_strings):
    edges = set()
    for s in bin_strings:
        low, high = s.split("-")
        edges.add(float(low))
        edges.add(float(high))
    return np.array(sorted(edges), dtype=float)


def _bin_low(s: str) -> float:
    return float(s.split("-")[0])


def extract_shell_to_shell_matrix(results, formalism: str, term: str, method: str):
    if formalism not in results:
        raise ValueError(f"Formalism '{formalism}' not found. Available: {list(results.keys())}")
    if term not in results[formalism]:
        raise ValueError(f"Term '{term}' not found. Available: {list(results[formalism].keys())}")
    if method not in results[formalism][term]:
        raise ValueError(f"Method '{method}' not found. Available: {list(results[formalism][term].keys())}")

    k_bins_dict = results[formalism][term][method]
    all_bins = list(k_bins_dict.keys())
    k_bin_strings = select_partition_bins(all_bins, strategy="max_high")
    if not k_bin_strings:
        k_bin_strings = _sorted_bin_strings(all_bins)
    k_bin_strings = filter_positive_bins(k_bin_strings)
    if not k_bin_strings:
        raise ValueError("No positive-k bins available for log-scale plotting.")

    # Assume Q bins match K bins for AnyToAny; fall back to keys of first selected K bin.
    first_k = k_bin_strings[0]
    q_candidates = list(k_bins_dict.get(first_k, {}).keys())
    if set(k_bin_strings).issubset(set(q_candidates)):
        q_bin_strings = list(k_bin_strings)
    else:
        q_bin_strings = _sorted_bin_strings(q_candidates)

    n = len(k_bin_strings)
    T = np.zeros((n, n), dtype=float)
    for i, k_str in enumerate(k_bin_strings):
        row = k_bins_dict[k_str]
        for j, q_str in enumerate(q_bin_strings):
            T[i, j] = float(row.get(q_str, 0.0))

    edges = edges_from_partition(k_bin_strings)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean (log bins)
    return T, edges, centers, k_bin_strings, q_bin_strings


def _bin_high(s: str) -> float:
    return float(s.split("-")[1])


def select_partition_bins(bin_strings, strategy: str = "max_high", start_low: float = None):
    """
    Many result files contain multiple/overlapping bin sets (e.g. from repeated runs).
    For plotting we often need a *single* partition of k-space: contiguous, non-overlapping bins.

    This builds a contiguous chain by following (low -> high) edges:
      current_low = start_low (default: min low)
      choose next bin among those with this low:
        - strategy='max_high': pick the widest bin (useful for log-like bins)
        - strategy='min_high': pick the narrowest bin
    """
    by_low = {}
    for s in bin_strings:
        low, high = map(float, s.split("-"))
        by_low.setdefault(low, []).append((high, s))

    if start_low is None:
        start_low = min(by_low.keys()) if by_low else None
    if start_low is None:
        return []

    out = []
    seen = set()
    cur = float(start_low)
    while cur in by_low:
        candidates = by_low[cur]
        if strategy == "min_high":
            high, s = min(candidates, key=lambda t: t[0])
        else:
            high, s = max(candidates, key=lambda t: t[0])
        if s in seen:
            break
        out.append(s)
        seen.add(s)
        cur = float(high)
    return out


def edges_from_partition(bin_strings):
    if not bin_strings:
        return np.array([], dtype=float)
    edges = [float(bin_strings[0].split("-")[0])]
    for s in bin_strings:
        edges.append(float(s.split("-")[1]))
    return np.array(edges, dtype=float)


def filter_positive_bins(bin_strings):
    """Drop bins that include non-positive k (cannot be shown on log axes)."""
    out = []
    for s in bin_strings:
        low, high = map(float, s.split("-"))
        if low > 0 and high > 0:
            out.append(s)
    return out


def compute_cross_scale_flux(term_anytoany: dict, bin_strings=None):
    """
    Replicates plots.ipynb (cell "CrossScale"):
      Pi(smallK) = sum_{K shells with K_low >= smallK} sum_{Q shells with Q_low < smallK} T(K,Q)
    where T(K,Q) is the AnyToAny shell-to-shell transfer.

    Returns arrays (k_edges_inner, Pi).
    """
    if bin_strings is None:
        bin_strings = select_partition_bins(term_anytoany.keys(), strategy="max_high")
        if not bin_strings:
            bin_strings = _sorted_bin_strings(term_anytoany.keys())
    k_bins = filter_positive_bins(list(bin_strings))
    if not k_bins:
        return np.array([], dtype=float), np.array([], dtype=float)
    edges = edges_from_partition(k_bins)
    smallKs = edges[1:-1]  # exclude outermost edges

    pi = []
    for smallK in smallKs:
        tmp = 0.0
        for k_bin in k_bins:
            k_low = _bin_low(k_bin)
            if k_low < smallK:
                continue
            row = term_anytoany[k_bin]
            for q_bin in k_bins:
                q_low = _bin_low(q_bin)
                if q_low >= smallK:
                    continue
                tmp += float(row.get(q_bin, 0.0))
        pi.append(tmp)

    return np.array(smallKs, dtype=float), np.array(pi, dtype=float)


def mean_inertial_flux(k, pi, low: float, high: float):
    mask = (k >= low) & (k <= high)
    if not mask.any():
        return None
    return float(np.mean(pi[mask]))


def transfer_matrix_stats(T: np.ndarray):
    T = np.asarray(T, dtype=float)
    out = {}
    out["shape"] = tuple(T.shape)
    out["finite"] = bool(np.isfinite(T).all())
    out["min"] = float(np.min(T)) if T.size else float("nan")
    out["max"] = float(np.max(T)) if T.size else float("nan")
    out["maxabs"] = float(np.max(np.abs(T))) if T.size else float("nan")
    out["sum"] = float(np.sum(T)) if T.size else float("nan")
    diag = np.diag(T) if T.ndim == 2 else np.array([])
    out["diag_mean"] = float(np.mean(diag)) if diag.size else float("nan")
    out["diag_maxabs"] = float(np.max(np.abs(diag))) if diag.size else float("nan")
    if T.ndim == 2 and T.shape[0] == T.shape[1] and T.size:
        out["antisym_maxabs"] = float(np.max(np.abs(T + T.T)))
    else:
        out["antisym_maxabs"] = float("nan")
    if T.ndim == 2 and T.size:
        net_into_K = np.sum(T, axis=1)  # sum over Q
        out["net_into_K_sum"] = float(np.sum(net_into_K))
        out["net_into_K_maxabs"] = float(np.max(np.abs(net_into_K)))
    else:
        out["net_into_K_sum"] = float("nan")
        out["net_into_K_maxabs"] = float("nan")
    return out


def print_term_stats(term: str, stats: dict):
    print(f"\n[{term}]")
    print(f"  shape              = {stats['shape']}")
    print(f"  finite             = {stats['finite']}")
    print(f"  min/max            = {stats['min']:.6e} / {stats['max']:.6e}")
    print(f"  max |T|            = {stats['maxabs']:.6e}")
    print(f"  sum(T)             = {stats['sum']:.6e}   (≈0 only if bins cover all interacting scales)")
    print(f"  max |T+T^T|        = {stats['antisym_maxabs']:.6e}   (antisymmetry diagnostic)")
    print(f"  diag mean/max|diag|= {stats['diag_mean']:.6e} / {stats['diag_maxabs']:.6e}")
    print(f"  sum_K sum_Q T(K,Q) = {stats['net_into_K_sum']:.6e}   (matches sum(T))")
    print(f"  max |sum_Q T(K,Q)| = {stats['net_into_K_maxabs']:.6e}   (net into each K shell)")


def plot_heatmap(ax, T, edges, term: str, norm_by: float = None, cmap="RdBu_r"):
    data = T.copy()
    cbar_label = r"$\mathcal{T}(Q,K)$"
    if norm_by is not None and norm_by != 0:
        data = data / norm_by
        cbar_label = r"$\mathcal{T}(Q,K) / \mathcal{N}$"

    lim = np.max(np.abs(data)) if data.size else 1.0
    if lim == 0:
        lim = 1.0
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    im = ax.pcolormesh(edges, edges, data, cmap=cmap, norm=norm, shading="flat", rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wavenumber $Q$")
    ax.set_ylabel("wavenumber $K$")
    ax.set_title(term)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(edges[0], edges[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], "k--", alpha=0.3, lw=1)
    return im, cbar_label


def main():
    ap = argparse.ArgumentParser(description="Plot hydro energy transfer (heatmaps + flux) from transfer_output.pkl")
    ap.add_argument("pickle_file", help="Pickle file containing transfer analysis results")
    ap.add_argument("--formalism", default="WW", help="Formalism (default: WW)")
    ap.add_argument("--method", default="AnyToAny", help="Method (default: AnyToAny)")
    ap.add_argument("--terms", nargs="+", default=None, help="Terms to plot (default: UU UUA UUC if present)")
    ap.add_argument("--stats", action="store_true", help="Print sanity-check stats for each plotted term")
    ap.add_argument("--check-uu-sum", action="store_true", help="Check that UU ≈ UUA + UUC (if all are available)")
    ap.add_argument("--no-heatmap", action="store_true", help="Do not plot shell-to-shell heatmaps")
    ap.add_argument("--no-flux", action="store_true", help="Do not plot cross-scale flux")
    ap.add_argument("--inertial-range", type=float, nargs=2, default=(7.0, 16.0), metavar=("LOW", "HIGH"),
                    help="Inertial-range bounds for mean flux (default: 7 16)")
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="Alias for --normalize-by eps0 (kept for backwards compatibility).",
    )
    ap.add_argument(
        "--normalize-by",
        default="none",
        choices=["none", "eps0", "uu_max"],
        help="Normalization for heatmaps/flux: none | eps0 (mean inertial flux of UU) | uu_max (max|T_UU|).",
    )
    ap.add_argument("--cmap", default="RdBu_r", help="Colormap for heatmaps (default: RdBu_r)")
    ap.add_argument("--output", default=None, help="Output PNG filename (default: auto)")
    ap.add_argument("--save", action="store_true", help="Save the figure")
    ap.add_argument("--show-plot", action="store_true", help="Show plot interactively")
    ap.add_argument("--dpi", type=int, default=200, help="DPI for saved PNG (default: 200)")
    args = ap.parse_args()

    results = load_transfer_data(args.pickle_file)
    if args.formalism not in results:
        raise SystemExit(f"Formalism '{args.formalism}' not found. Available: {list(results.keys())}")

    available_terms = list(results[args.formalism].keys())
    if args.terms is None:
        preferred = ["UU", "UUA", "UUC"]
        terms = [t for t in preferred if t in available_terms]
        if not terms:
            terms = sorted(available_terms)
    else:
        terms = args.terms

    low, high = args.inertial_range
    if args.normalize and args.normalize_by == "none":
        args.normalize_by = "eps0"

    norm_scalar = None
    norm_label = None
    if args.normalize_by != "none":
        if "UU" not in results[args.formalism] or args.method not in results[args.formalism]["UU"]:
            print("[WARN] Requested normalization, but UU data not available; skipping normalization.")
        elif args.normalize_by == "eps0":
            uu_any = results[args.formalism]["UU"][args.method]
            uu_bins = select_partition_bins(uu_any.keys(), strategy="max_high")
            k, pi = compute_cross_scale_flux(uu_any, bin_strings=uu_bins)
            norm_scalar = mean_inertial_flux(k, pi, low=low, high=high)
            norm_label = rf"$\epsilon_0$ (mean $\Pi_{{UU}}(k)$, {low:g}..{high:g})"
        elif args.normalize_by == "uu_max":
            Tuu, _, _, _, _ = extract_shell_to_shell_matrix(results, args.formalism, "UU", args.method)
            norm_scalar = float(np.max(np.abs(Tuu))) if Tuu.size else None
            norm_label = r"$\max|\mathcal{T}_{UU}(Q,K)|$"

        if norm_scalar in (None, 0) or not np.isfinite(norm_scalar):
            print("[WARN] Normalization scalar is invalid; skipping normalization.")
            norm_scalar = None
            norm_label = None

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.pickle_file))[0]
        args.output = f"{base}_hydro_transfer.png"

    do_heatmap = not args.no_heatmap
    do_flux = not args.no_flux
    if not do_heatmap and not do_flux and not (args.stats or args.check_uu_sum):
        raise SystemExit("Nothing to do: set --stats/--check-uu-sum or enable plotting.")

    if args.stats:
        collected = {}
        for term in terms:
            if term not in results[args.formalism] or args.method not in results[args.formalism][term]:
                continue
            T, _, _, _, _ = extract_shell_to_shell_matrix(results, args.formalism, term, args.method)
            st = transfer_matrix_stats(T)
            collected[term] = st
            print_term_stats(term, st)

        if "UU" in results[args.formalism] and args.method in results[args.formalism]["UU"]:
            uu_any = results[args.formalism]["UU"][args.method]
            bins = select_partition_bins(uu_any.keys(), strategy="max_high")
            k, pi = compute_cross_scale_flux(uu_any, bin_strings=bins)
            eps = mean_inertial_flux(k, pi, low=low, high=high) if k.size else None
            if eps is not None:
                print(f"\n[UU] mean Π(k) over inertial range {low:g}..{high:g} = {eps:.6e}")
        if norm_scalar is not None and norm_label is not None:
            print(f"\n[normalization] {args.normalize_by} => {norm_label} = {norm_scalar:.6e}")

        if "UUA" in collected and "UUC" in collected:
            a = collected["UUA"]["maxabs"]
            c = collected["UUC"]["maxabs"]
            if np.isfinite(a) and a != 0:
                print(f"\n[UUC vs UUA] max|UUC|/max|UUA| = {c/a:.3e}")
        if "UU" in collected and "UUC" in collected:
            u = collected["UU"]["maxabs"]
            c = collected["UUC"]["maxabs"]
            if np.isfinite(u) and u != 0:
                print(f"[UUC vs UU]  max|UUC|/max|UU|  = {c/u:.3e}")

    if args.check_uu_sum:
        have = all(t in results[args.formalism] and args.method in results[args.formalism][t] for t in ("UU", "UUA", "UUC"))
        if not have:
            print("\n[check-uu-sum] Need UU, UUA, UUC present to check.")
        else:
            Tuu, _, _, _, _ = extract_shell_to_shell_matrix(results, args.formalism, "UU", args.method)
            Ta, _, _, _, _ = extract_shell_to_shell_matrix(results, args.formalism, "UUA", args.method)
            Tc, _, _, _, _ = extract_shell_to_shell_matrix(results, args.formalism, "UUC", args.method)
            diff = Tuu - (Ta + Tc)
            print("\n[check-uu-sum] UU - (UUA + UUC):")
            print(f"  max |diff| = {np.max(np.abs(diff)):.6e}")
            print(f"  mean|diff| = {np.mean(np.abs(diff)):.6e}")

    n_heat = len(terms) if do_heatmap else 0
    nrows = (1 if do_heatmap else 0) + (1 if do_flux else 0)
    if do_heatmap and do_flux:
        fig = plt.figure(figsize=(4.2 * max(1, n_heat), 7.0), constrained_layout=True)
        gs = fig.add_gridspec(2, max(1, n_heat), height_ratios=[1.0, 0.7])
    elif do_heatmap:
        fig = plt.figure(figsize=(4.2 * max(1, n_heat), 4.2), constrained_layout=True)
        gs = fig.add_gridspec(1, max(1, n_heat))
    else:
        fig = plt.figure(figsize=(6.5, 3.8), constrained_layout=True)
        gs = fig.add_gridspec(1, 1)

    ims = []
    cbar_label = None

    if do_heatmap:
        for i, term in enumerate(terms):
            ax = fig.add_subplot(gs[0, i] if do_flux else gs[0, i])
            T, edges, _, _, _ = extract_shell_to_shell_matrix(results, args.formalism, term, args.method)
            im, lbl = plot_heatmap(ax, T, edges, term=term, norm_by=norm_scalar, cmap=args.cmap)
            ims.append(im)
            cbar_label = lbl

        # single colorbar for all heatmaps
        if ims:
            fig.colorbar(ims[0], ax=[ax for ax in fig.axes[:n_heat]], orientation="horizontal", fraction=0.05, pad=0.08,
                         label=cbar_label)

    if do_flux:
        axf = fig.add_subplot(gs[1, :] if do_heatmap else gs[0, 0])
        for term in terms:
            if term not in results[args.formalism] or args.method not in results[args.formalism][term]:
                continue
            term_any = results[args.formalism][term][args.method]
            bins = select_partition_bins(term_any.keys(), strategy="max_high")
            k, pi = compute_cross_scale_flux(term_any, bin_strings=bins)
            y = pi
            label = term
            if norm_scalar not in (None, 0):
                y = y / norm_scalar
                label = f"{term}/{args.normalize_by}"
            axf.plot(k, y, label=label)

        axf.axvspan(low, high, color="k", alpha=0.06, label="inertial range")
        axf.set_xscale("log")
        axf.grid(True, which="both", alpha=0.3)
        axf.set_xlabel("wavenumber $k$")
        axf.set_ylabel(r"cross-scale flux $\Pi(k)$" + (r" / $\mathcal{N}$" if norm_scalar not in (None, 0) else ""))
        axf.legend(fontsize=9, loc="best")

        if norm_scalar is not None and norm_label is not None:
            axf.text(
                0.98,
                0.02,
                rf"$\mathcal{{N}}$={norm_scalar:.3e} ({args.normalize_by})",
                transform=axf.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
            )

    if args.save:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Wrote {args.output}")
    if args.show_plot:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
