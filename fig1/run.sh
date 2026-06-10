nvcc -O3 -arch=native coupled_elastic_chains.cu -o coupled_chains_sim

for((seed=1234;seed<1394;seed++))
do 
	./coupled_chains_sim $seed; 
	mv displacement_spectra.dat "displacement_spectra_"$seed".dat"
	mv bragg_glass_thermal_output.dat "bragg_glass_thermal_output_"$seed".dat"
done
