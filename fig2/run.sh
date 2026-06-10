nvcc -O3 -arch=native -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/12.3/include/ \
coupled_elastic_chains.cu -DHARDCORE -o coupled_chains_sim -lcurand

a=$(echo "1234+$1" | bc -l)

for((seed=1234;seed<$a;seed++))
do 
	./coupled_chains_sim $seed; 
	mv displacement_spectra.dat "displacement_spectra_"$seed".dat"
	mv bragg_glass_thermal_output.dat "bragg_glass_thermal_output_"$seed".dat"
done
