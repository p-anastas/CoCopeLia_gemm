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
make clean
cmake ../
make -j 8
cd $PROJECT_DIR

RESDIR=$PROJECT_DIR/BenchOutputs/$machine

mkdir -p "${RESDIR}/microbench_logs"

transfer_log="${RESDIR}/microbench_logs/transfer_microbench_gpu.log"
dgemm_log="${RESDIR}/microbench_logs/dgemm_microbench_gpu.log"
sgemm_log="${RESDIR}/microbench_logs/sgemm_microbench_gpu.log"
dgemv_log="${RESDIR}/microbench_logs/dgemv_microbench_gpu.log"
daxpy_log="${RESDIR}/microbench_logs/daxpy_microbench_gpu.log"

micro_transfer_exec="${BUILDIR}/transfers_microbench_gpu"
micro_dgemm_exec="${BUILDIR}/dgemm_microbench_gpu"
micro_sgemm_exec="${BUILDIR}/sgemm_microbench_gpu"
micro_dgemv_exec="${BUILDIR}/dgemv_microbench_gpu"
micro_daxpy_exec="${BUILDIR}/daxpy_microbench_gpu"

# General benchmark steps
step=256

#Transfer Tile size
Transfers_minDim=256
Transfers_maxDim=16384
echo "Performing microbenchmarks for transfers..."
rm $transfer_log
echo "$micro_transfer_exec $machine $device -1 $Transfers_minDim $Transfers_maxDim $step &>> $transfer_log"
$micro_transfer_exec $machine $device -1 $Transfers_minDim $Transfers_maxDim $step &>> $transfer_log
echo "$micro_transfer_exec $machine -1 $device $Transfers_minDim $Transfers_maxDim $step &>> $transfer_log"
$micro_transfer_exec $machine -1 $device $Transfers_minDim $Transfers_maxDim $step &>> $transfer_log
echo "Done"

# dgemm micro-benchmark Tile size
dgemm_minDim=256
dgemm_maxDim=16384
echo "Performing microbenchmarks for dgemm..."
rm $dgemm_log
echo "$micro_dgemm_exec $machine $device $dgemm_minDim $dgemm_maxDim $step &>> $dgemm_log"
$micro_dgemm_exec $machine $device $dgemm_minDim $dgemm_maxDim $step &>> $dgemm_log
echo "Done"

# sgemm micro-benchmark Tile size
sgemm_minDim=256
sgemm_maxDim=16384

echo "Performing microbenchmarks for sgemm..."
rm $sgemm_log
echo "$micro_sgemm_exec $machine $device $sgemm_minDim $sgemm_maxDim  $step &>> $sgemm_log"
$micro_sgemm_exec $machine $device $sgemm_minDim $sgemm_maxDim $step &>> $sgemm_log
echo "Done"

# gemv micro-benchmark deployment values
dgemv_minDim=256
dgemv_maxDim=16384

echo "Performing microbenchmarks for dgemv..."
rm $dgemv_log
echo "$micro_dgemv_exec $machine $device $dgemv_minDim $dgemv_maxDim $step &>> $dgemv_log"
$micro_dgemv_exec $machine $device $dgemv_minDim $dgemv_maxDim $step &>> $dgemv_log
echo "Done"

# axpy micro-benchmark deployment values
step=1048576
daxpy_minDim=1048576
daxpy_maxDim=268435456 #(16384^2)

echo "Performing microbenchmarks for daxpy..."
rm $daxpy_log
echo "$micro_daxpy_exec $machine $daxpy_minDim $device $daxpy_maxDim $step &>> $daxpy_log"
$micro_daxpy_exec $machine $device $daxpy_minDim $daxpy_maxDim $step &>> $daxpy_log
echo "Done"
