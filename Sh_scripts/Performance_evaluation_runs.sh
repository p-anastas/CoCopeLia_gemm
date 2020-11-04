#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

PROJECT_DIR=/home/users/panastas/PhD_stuff/CoCopeLia
cd $PROJECT_DIR

BUILDIR=$1
machine=$2
device=$3

## Automatically build the microbenchmarks every time
cp CMakeLists_${machine}.txt CMakeLists.txt
cd $BUILDIR
cmake ../
make clean
make -j 8
cd $PROJECT_DIR

RESDIR=$PROJECT_DIR/Data_manipulation/Results/$machine

#mkdir -p ${RESDIR}
mkdir -p "${RESDIR}/exec_logs"
mkdir -p "${RESDIR}/evaluation"

rm ${RESDIR}/exec_logs/*_perf_eval.log
dgemm_perf_log="${RESDIR}/exec_logs/Dgemm_perf_eval.log"
sgemm_perf_log="${RESDIR}/exec_logs/sgemm_perf_eval.log"
daxpy_perf_log="${RESDIR}/exec_logs/daxpy_perf_eval.log"

for FUNC in Dgemm Sgemm 
do
CoCopelia_run="$BUILDIR/CoCopeLia_${FUNC}"
cuBLASXt_run="$BUILDIR/cuBLASXt_${FUNC}"
BLASX_run="$BUILDIR/BLASX_${FUNC}"
Serial_run="$BUILDIR/Serial_${FUNC}"

echo "Performing Benchmarks for CoCodDgemm evaluation..."
for A_loc in 1 0;
do
	for B_loc in 1 0;
	do
		for C_loc in 1 0;
		do
			for Sq in {4096..16384..1024}
			do 
				echo "$CoCopelia_run $machine $Sq $Sq $Sq -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log"
				$CoCopelia_run $machine $Sq $Sq $Sq -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log
				echo "$cuBLASXt_run $machine $Sq $Sq $Sq -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log"
				$cuBLASXt_run $machine $Sq $Sq $Sq -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log
				echo "$BLASX_run $machine $Sq $Sq $Sq -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log"
				$BLASX_run $machine $Sq $Sq $Sq -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log
				echo "$Serial_run $machine $Sq $Sq $Sq $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log"
				$Serial_run $machine $Sq $Sq $Sq $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log
			done


		done 
	done
done
A_loc=1
B_loc=1
C_loc=1
for inbalanced in {4096..16384..1024}
do 
	for ctr in 3 4 5 #2 3 4 5 6 7 8; # testbedI for 12000 can't do 7,8
	do 
		fat=$(($inbalanced*$ctr/2))
		double_thin=$(($inbalanced*4/$ctr/$ctr))

		echo "$CoCopelia_run $machine $fat $fat $double_thin -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log"
		$CoCopelia_run $machine $fat $fat $double_thin -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log
		echo "$BLASX_run $machine $fat $fat $double_thin -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log"
		$BLASX_run $machine $fat $fat $double_thin -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log
		echo "$cuBLASXt_run $machine $fat $fat $double_thin -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log"
		$cuBLASXt_run $machine $fat $fat $double_thin -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log
	done

	for ctr in 3 4 5 #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16;
	do
		double_fat=$(($inbalanced*$ctr*$ctr/4))
		thin=$(($inbalanced*2/$ctr))

		echo "$CoCopelia_run $machine $thin $thin $double_fat -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log"
		$CoCopelia_run $machine $thin $thin $double_fat -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log
		echo "$BLASX_run $machine $thin $thin $double_fat -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log"
		$BLASX_run $machine $thin $thin $double_fat -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log
		echo "$cuBLASXt_run $machine $thin $thin $double_fat -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log"
		$cuBLASXt_run $machine $thin $thin $double_fat -1 $device $A_loc $B_loc $C_loc &>> $dgemm_perf_log
	done
done
echo "Done"
done
exit 1

Daxpy_benchmark="$BUILDIR/Daxpy_benchmark"


echo "Performing Benchmarks for CoCodaxpy evaluation..."
for x_loc in 1 0;
do
	for y_loc in 1 0;
	do

		for N in 16384 #{16384..1638400..16384}
		do 
			for mult in 64 128 256 512 1024 2048 4096 8192 16384
			do 
				echo "$Daxpy_benchmark $machine $(($N*$mult)) $device $x_loc $y_loc &>> $daxpy_perf_log"
				$Daxpy_benchmark $machine $(($N*$mult)) $device $x_loc $y_loc &>> $daxpy_perf_log
			done
		done

	done
done
echo "Done"





