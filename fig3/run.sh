nvcc -O3 -arch=native -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/12.3/include/ \
coupled_elastic_chains.cu -DHARDCORE -o coupled_chains_sim -lcurand

a=$(echo "1234+$1" | bc -l)

for((seed=1234;seed<$a;seed++))
do 
	./coupled_chains_sim $seed --params params.ini 

	for file in displacement_spectra_replica_*.dat
	do
		mv $file "seed_"$seed"_"$file 
	done

	for file in correlation_replica_*.dat
	do
		mv $file "seed_"$seed"_"$file 
	done
done
