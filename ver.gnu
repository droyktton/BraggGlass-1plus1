set multi lay 2,1

set tit 'Nz=32, Nx=1024'

set key left
set logs

set xla 'q'
set yla 'q^2 <|u(q)|^2>'

plot [:5][:] "< cat displacement_spectra_*.dat | sort -n -k 1" u 2:($2**2*$3*1024) smooth un w lp t 'sim', \
0.5*x/2.0 t 'Thermal \~q', 1e-3/x**2 t 'Larkin \~1/q^2'


unset tit
set xla 'r'
set yla 'B(r)'

plot "< cat bragg_glass_thermal_output_*.dat | sort -n -k 1" u 1:2 smooth un w lp t 'sim',\
.25*x t '(zeta=1/2) Thermal \~x', 0.04*x**2 t '(zeta=3/2) Larkin \~x^{2}' 

unset multi
