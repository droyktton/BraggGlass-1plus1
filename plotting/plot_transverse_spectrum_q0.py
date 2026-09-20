#!/usr/bin/env python3
"""Plot the vortex-lattice density structure factor near Q=0: S_rho(q) =
4*Nx*sin^2(q/2) * S_u^(x)(q), an exact rescaling of
transverse_spectrum_replica_<T>.dat (the spectrum of u along x -- across
chains -- not the along-chain S_u(qy)). See plot_structure_factor_q0.py
for the along-y analogue and the derivation of the Nx factor.

Usage:
    python plotting/plot_transverse_spectrum_q0.py --nx 32 [--dir .] [--out plots/transverse_spectrum_q0.png] [--show]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from lib import SURFACE, plot_transverse_spectrum_q0, style_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".",
                     help="directory to search recursively for *_transverse_spectrum_replica_*.dat")
    ap.add_argument("--nx", type=int, required=True, help="Nx used for the run (params.ini)")
    ap.add_argument("--out", default="plots/transverse_spectrum_q0.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=SURFACE)
    if not plot_transverse_spectrum_q0(ax, args.dir, args.nx):
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
