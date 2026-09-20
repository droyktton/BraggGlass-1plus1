#!/usr/bin/env python3
"""Plot the exact density (Bragg-peak) structure factor S_rho(qy), with an
optional Debye-Waller estimate overlaid from the matching B(r) file so you
can see where the harmonic approximation departs from the exact result.

Usage:
    python plotting/plot_structure_factor.py --ny 64 [--dir .] [--out plots/structure_factor.png] [--show]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from lib import SURFACE, plot_structure_factor, style_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".",
                     help="directory to search recursively for *_structure_factor_replica_*.dat")
    ap.add_argument("--ny", type=int, required=True, help="Ny used for the run (params.ini)")
    ap.add_argument("--out", default="plots/structure_factor.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=SURFACE)
    if not plot_structure_factor(ax, args.dir, args.ny):
        raise SystemExit(f"No structure_factor_replica_*.dat files found under {args.dir}")
    style_axes(ax, log_x=False, log_y=True)
    ax.legend(frameon=False, fontsize=8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
