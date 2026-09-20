#!/usr/bin/env python3
"""Plot the real-space displacement correlations B_y(r) and B_x(r) against
the thermal and Larkin roughness-exponent predictions.

Usage:
    python plotting/plot_correlation.py [--dir .] [--out plots/correlation.png] [--show]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from lib import SURFACE, plot_correlation, style_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".",
                     help="directory to search recursively for *_correlation_replica_*.dat")
    ap.add_argument("--out", default="plots/correlation.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    if not plot_correlation(ax, args.dir):
        raise SystemExit(f"No correlation_replica_*.dat files found under {args.dir}")
    style_axes(ax, log_x=True, log_y=True)
    # Up to 2 series (B_y, B_x) per temperature plus 2 reference curves adds
    # up fast -- no inside corner stays clean, so this legend goes outside.
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
