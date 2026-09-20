# Diagnostic plots for a seed scan produced by run.sh (or slurm/run_array.slurm).
# Edit Nx, Ny below to match the params.ini used for the run before plotting.
# For anything beyond these three quick panels (multi-temperature overlays,
# the Debye-Waller estimate vs. the exact vortex-lattice S_rho, the
# along-chain-only S_rho variants, saved PNG/PDF), use the scripts in
# plotting/ instead -- see plotting/plot_all.py.
Nx = 32
Ny = 64

set multi lay 3,1

set tit sprintf('Nx=%d, Ny=%d', Nx, Ny)

set key left
set logs

set xla 'q'
set yla 'q^2 <|u(q)|^2>'

plot [:5][:] "< cat seed_*_displacement_spectra_replica_*.dat | sort -n -k 1" u 2:($2**2*$3*Ny) smooth un w lp t 'sim', \
0.5*x/2.0 t 'Thermal \~q', 1e-3/x**2 t 'Larkin \~1/q^2'


unset tit
unset logs x
set xla 'q'
set yla 'S_{rho}(q), vortex lattice'
set arrow from pi,graph 0 to pi,graph 1 nohead lt 0

# Vortex-LATTICE density structure factor: u(x,y) is a displacement along x
# (see the HARDCORE constraint in the kernel, which compares u against its
# x-neighbors at the same y to keep chains from crossing), so translational
# order of the Nx chains -- the actual Bragg-glass order -- lives in
# Fourier transforms along x, not y (S_u(qy) above measures a DIFFERENT
# thing: single-chain roughness along its own length).
#
# Full first Brillouin zone, q in [0,2pi], stitching two branches at q=pi
# (a DFT on integer x only resolves q modulo 2pi, so these tile the zone
# exactly once): q in [0,pi] is the Q=0/compressional approximation, an
# exact algebraic rescaling of transverse_spectrum (S_rho = |e^{iq}-1|^2
# S_u^(x) = 4 sin^2(q/2) S_u^(x) -> q^2 S_u^(x) as q->0, not a separate
# simulation output); q in [pi,2pi] is the exact Bragg-peak measurement via
# q = 2*pi - qx.
plot "< cat seed_*_transverse_spectrum_replica_*.dat | sort -n -k 1" u 2:(4*sin($2/2)**2*$3) smooth un w lp t 'Q=0 (approx, from S_u^{(x)})', \
"< cat seed_*_transverse_structure_factor_replica_*.dat | sort -n -k 1" u (2*pi-$2):3 smooth un w lp t 'Bragg peak Q=2pi (exact)'

unset arrow
set logs x
set xla 'r'
set yla 'B(r)'

plot "< cat seed_*_correlation_replica_*.dat | sort -n -k 1" u 1:2 smooth un w lp t 'sim',\
.25*x t '(zeta=1/2) Thermal \~x', 0.04*x**2 t '(zeta=3/2) Larkin \~x^{2}'

unset multi
