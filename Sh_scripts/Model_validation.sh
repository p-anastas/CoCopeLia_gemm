#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

PROJECT_DIR=/home/users/panastas/PhD_stuff/CoCopeLia
cd $PROJECT_DIR

BUILDIR=$1
machine=$2
device=$3
version=$4

## Automatically build the microbenchmarks every time
cp CMakeLists_${machine}.txt CMakeLists.txt
cd $BUILDIR
make clean
cmake ../
make -j 8
cd $PROJECT_DIR

RESDIR=$PROJECT_DIR/Data_manipulation/Results/$machine

#mkdir -p ${RESDIR}
mkdir -p "${RESDIR}/exec_logs"
mkdir -p "${RESDIR}/validation"

CoCopelia_pred="$BUILDIR/CoCopeLia_predict"

for FUNC in Dgemm Sgemm
do
	echo "Generating predictions for CoCopelia ${FUNC} validation..."
	rm ${RESDIR}/validation/${FUNC}_CoCopelia_predict_*_${device}_v$version.log
	test_log="${RESDIR}/exec_logs/${FUNC}_val_pred.log"
	rm $test_log
	for A_loc in 1 0;
	do
		for B_loc in 1 0;
		do
			for C_loc in 1 0;
			do

			for Sq in {4096..16384..4096}
			do 
				echo "$CoCopelia_pred $machine $Sq $Sq $Sq $device $FUNC $A_loc $B_loc $C_loc &>> $test_log"
				$CoCopelia_pred $machine $Sq $Sq $Sq $device $FUNC $A_loc $B_loc $C_loc &>> $test_log
			done

			done 
		done
	done
	A_loc=1
	B_loc=1
	C_loc=1
	for inbalanced in {4096..16384..4096}
	do 
		for ctr in 3 4 5 #2 3 4 5 6 7 8; # testbedI for 12000 can't do 7,8
		do 
			fat=$(($inbalanced*$ctr/2))
			double_thin=$(($inbalanced*4/$ctr/$ctr))

			echo "$CoCopelia_pred $machine $fat $fat $double_thin $device $FUNC $A_loc $B_loc $C_loc &>> $test_log"
			$CoCopelia_pred $machine $fat $fat $double_thin $device $FUNC $A_loc $B_loc $C_loc &>> $test_log
		done
	
		for ctr in 3 4 5; #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16;
		do
			double_fat=$(($inbalanced*$ctr*$ctr/4))
			thin=$(($inbalanced*2/$ctr))

			echo "$CoCopelia_pred $machine $thin $thin $double_fat $device $FUNC $A_loc $B_loc $C_loc &>> $test_log"
			$CoCopelia_pred $machine $thin $thin $double_fat $device $FUNC $A_loc $B_loc $C_loc &>> $test_log
		done
	done
	echo "${FUNC} predictions Done"
done 

for FUNC in Dgemm Sgemm 
do
	CoCopelia_run="$BUILDIR/CoCopeLia_${FUNC}"
	cuBLASXt_run="$BUILDIR/cuBLASXt_${FUNC}"
	test_log="${RESDIR}/exec_logs/${FUNC}_val_set.log"
	rm $test_log
	echo "Generating set for CoCopelia ${FUNC} validation..."
	for A_loc in 1 0;
	do
		for B_loc in 1 0;
		do
			for C_loc in 1 0;
			do
				for T in {512..16384..256}
				do 
			
					for Sq in {4096..16384..4096}
					do 
						echo "$CoCopelia_run $machine $Sq $Sq $Sq $T $device $A_loc $B_loc $C_loc &>> $test_log"
						$CoCopelia_run $machine $Sq $Sq $Sq $T $device $A_loc $B_loc $C_loc &>> $test_log
						echo "$cuBLASXt_run $machine $Sq $Sq $Sq $T $device $A_loc $B_loc $C_loc &>> $test_log"
						$cuBLASXt_run $machine $Sq $Sq $Sq $T $device $A_loc $B_loc $C_loc &>> $test_log
					done

				done

			done 
		done
	done
	A_loc=1
	B_loc=1
	C_loc=1
	for T in {512..16384..256}
	do 
		for inbalanced in {4096..16384..4096}
		do 
			for ctr in 3 4 5 #2 3 4 5 6 7 8; # testbedI for 12000 can't do 7,8
			do 
				fat=$(($inbalanced*$ctr/2))
				double_thin=$(($inbalanced*4/$ctr/$ctr))

				echo "$CoCopelia_run $machine $fat $fat $double_thin $T $device $A_loc $B_loc $C_loc &>> $test_log"
				$CoCopelia_run $machine $fat $fat $double_thin $T $device $A_loc $B_loc $C_loc &>> $test_log
				echo "$cuBLASXt_run $machine $fat $fat $double_thin $T $device $A_loc $B_loc $C_loc &>> $test_log"
				$cuBLASXt_run $machine $fat $fat $double_thin $T $device $A_loc $B_loc $C_loc &>> $test_log
			done
	
			for ctr in 3 4 5; #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16;
			do
				double_fat=$(($inbalanced*$ctr*$ctr/4))
				thin=$(($inbalanced*2/$ctr))

				echo "$CoCopelia_run $machine $thin $thin $double_fat $T $device $A_loc $B_loc $C_loc &>> $test_log"
				$CoCopelia_run $machine $thin $thin $double_fat $T $device $A_loc $B_loc $C_loc &>> $test_log
				echo "$cuBLASXt_run $machine $thin $thin $double_fat $T $device $A_loc $B_loc $C_loc &>> $test_log"
				$cuBLASXt_run $machine $thin $thin $double_fat $T $device $A_loc $B_loc $C_loc &>> $test_log
			done
		done
	done

done
echo "Done"

