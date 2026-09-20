#!/usr/bin/env python3
"""Plot the y-averaged transverse displacement spectrum S_u^(x)(qx) --
Nx*q^2*<|u(q)|^2> vs q -- against the thermal and Larkin scaling
predictions. This is the transverse analogue of plot_spectrum.py, and (with
the Nx factor included) the small-q "hydrodynamic" approximation to the
exact S_rho(qx) in plot_transverse_structure_factor.py -- plot both and
compare directly, or see plot_transverse_structure_factor_full.py which
already overlays this same curve on the exact one.

Usage:
    python plotting/plot_transverse_spectrum.py --nx 32 [--dir .] [--out plots/transverse_spectrum.png] [--show]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from lib import SURFACE, plot_transverse_spectrum, style_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".",
                     help="directory to search recursively for *_transverse_spectrum_replica_*.dat")
    ap.add_argument("--nx", type=int, required=True, help="Nx used for the run (params.ini)")
    ap.add_argument("--out", default="plots/transverse_spectrum.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=SURFACE)
    if not plot_transverse_spectrum(ax, args.dir, args.nx):
        raise SystemExit(f"No transverse_spectrum_replica_*.dat files found under {args.dir}")
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
