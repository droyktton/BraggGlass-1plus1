#!/usr/bin/env python3
"""Plot the chain-averaged displacement spectrum S_u(qy) against the
thermal and Larkin scaling predictions.

Usage:
    python plotting/plot_spectrum.py --ny 64 [--dir .] [--out plots/spectrum.png] [--show]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from lib import SURFACE, plot_spectrum, style_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".",
                     help="directory to search recursively for *_displacement_spectra_replica_*.dat")
    ap.add_argument("--ny", type=int, required=True, help="Ny used for the run (params.ini)")
    ap.add_argument("--out", default="plots/spectrum.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=SURFACE)
    if not plot_spectrum(ax, args.dir, args.ny):
        raise SystemExit(f"No displacement_spectra_replica_*.dat files found under {args.dir}")
    style_axes(ax, log_x=True, log_y=True)
    ax.legend(frameon=False, fontsize=8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
