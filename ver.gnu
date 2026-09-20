# Diagnostic plots for a seed scan produced by run.sh (or slurm/run_array.slurm).
# Edit Nx, Ny below to match the params.ini used for the run before plotting.
# For anything beyond these three quick panels (multi-temperature overlays,
# the Debye-Waller estimate vs. the exact S_rho, saved PNG/PDF), use the
# scripts in plotting/ instead -- see plotting/plot_all.py.
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
set yla 'S_{rho}(q)'

plot "< cat seed_*_structure_factor_replica_*.dat | sort -n -k 1" u 2:3 smooth un w lp t 'sim (exact)'

set logs x
set xla 'r'
set yla 'B(r)'

plot "< cat seed_*_correlation_replica_*.dat | sort -n -k 1" u 1:2 smooth un w lp t 'sim',\
.25*x t '(zeta=1/2) Thermal \~x', 0.04*x**2 t '(zeta=3/2) Larkin \~x^{2}'

unset multi
