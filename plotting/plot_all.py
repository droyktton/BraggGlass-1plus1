#!/usr/bin/env python3
"""All three diagnostics (S_u, S_rho, B(r)) in one figure -- the Python
equivalent of ver.gnu, plus the exact density structure factor.

Usage:
    python plotting/plot_all.py --ny 64 [--dir .] [--out plots/diagnostics.png] [--show]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from lib import SURFACE, plot_correlation, plot_spectrum, plot_structure_factor, style_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".",
                     help="directory to search recursively for *_replica_*.dat files")
    ap.add_argument("--ny", type=int, required=True, help="Ny used for the run (params.ini)")
    ap.add_argument("--out", default="plots/diagnostics.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor=SURFACE)

    found = False
    found |= plot_spectrum(axes[0], args.dir, args.ny)
    found |= plot_structure_factor(axes[1], args.dir, args.ny)
    found |= plot_correlation(axes[2], args.dir)
    if not found:
        raise SystemExit(f"No *_replica_*.dat files found under {args.dir}")

    style_axes(axes[0], log_x=True, log_y=True)
    style_axes(axes[1], log_x=False, log_y=True)
    style_axes(axes[2], log_x=True, log_y=True)
    # Data can fill the entire panel in any of the three (especially the
    # noisy exact S_rho), so legends go below each subplot -- an inside
    # corner risks covering data, and an outside-right legend on a middle
    # panel would land on top of its neighbor.
    for ax in axes:
        if ax.lines:
            ax.legend(frameon=False, fontsize=7, ncol=2,
                       loc="upper center", bbox_to_anchor=(0.5, -0.18))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
