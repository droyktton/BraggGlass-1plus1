set multiplot layout 2,1

# 1. Dynamically extract the unique replica values (floats) from the existing files
# This extracts the numbers, sorts them numerically, and formats them into a space-separated string.
replicas = system("ls seed_1234_displacement_spectra_replica_*.dat | grep -oP 'replica_\\K[0-9.]+(?=\.dat)' | sort -nu")

print replicas

set title 'Nz=32, Nx=512'
set key left
set logscale

set xlabel 'q'
set ylabel 'q^2 <|u(q)|^2>'

# 2. Loop over each float string token 'r' found in the 'replicas' variable
plot [:5][:] for [r in replicas] \
    sprintf("< cat seed_*_displacement_spectra_replica_%s.dat | sort -n -k 1", r) u 2:($2**2*$3*512) smooth unique w lp t sprintf('T = %s', r), \
    0.5*x/2.0 t 'Thermal ~q', 1e-3/x**2 t 'Larkin ~1/q^2'


unset title
set xlabel 'r'
set ylabel 'B(r)'

# The exact same loop logic applies to the second plot
plot for [r in replicas] \
    sprintf("< cat seed_*_correlation_replica_%s.dat | sort -n -k 1", r) u 1:2 smooth unique w lp t sprintf('T = %s', r), \
    .25*x t '(zeta=1/2) Thermal ~x', 0.04*x**2 t '(zeta=3/2) Larkin ~x^{2}'

unset multiplot
