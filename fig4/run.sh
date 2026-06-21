# solucion al problema de suspension:
#sudo rmmod nvidia_uvm
#sudo modprobe nvidia_uvm


CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda

nvcc -arch=sm_86 -L${CUDA_HOME}/lib64 --linker-options="-rpath,${CUDA_HOME}/lib64" \
-DHARDCORE -DNREPLICAS=10 \
-I/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/12.3/include/ \
-L/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/ \
coupled_elastic_chains.cu -lcurand -o coupled_chains_sim


#nvcc -O3 -arch=sm_86 -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/12.3/include/ \
#coupled_elastic_chains.cu -DHARDCORE -DNREPLICAS=5 -o coupled_chains_sim -lcurand

rm *.dat


a=$(echo "1234+$1" | bc -l)

for((seed=1234;seed<$a;seed++))
do 
	./coupled_chains_sim $seed 1234 --params params.ini 

	for file in displacement_spectra_replica_*.dat
	do
		mv $file "seed_"$seed"_"$file 
	done

	for file in correlation_replica_*.dat
	do
		mv $file "seed_"$seed"_"$file 
	done
done
