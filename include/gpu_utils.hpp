///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef GPU_UTILS_H
#define GPU_UTILS_H


#include <cuda.h>
#include "cublas_v2.h"

typedef struct gpu_timer {
  cudaEvent_t start;
  cudaEvent_t stop;
  float ms = 0;
} * gpu_timer_p;


typedef struct gpu_mempool {
  long long bytes, used_bytes;
  void * head_ptr, * avail_ptr;
short dev_id;
cudaStream_t pool_stream; 
} * mempool_p;

gpu_timer_p gpu_timer_init();
void gpu_timer_start(gpu_timer_p timer, cudaStream_t stream);
void gpu_timer_stop(gpu_timer_p timer, cudaStream_t stream);
float gpu_timer_get(gpu_timer_p timer);

/// Memory layout struct for matrices
enum mem_layout { ROW_MAJOR = 0, COL_MAJOR };

const char *print_mem(mem_layout mem);

/// Print name of loc for transfers
const char *print_loc(short loc);

/// Print all available CUDA devices and their basic info
void print_devices();

/// Check if there are CUDA errors on the stack
void cudaCheckErrors();

/// Allocate 'count' bytes of CUDA device memory (+errorcheck)
void* gpu_malloc(long long count);
/// Allocate 'count' bytes of CUDA host pinned memory (+errorcheck)
void* pin_malloc(long long count);
/// Generalized malloc in loc 
void vec_alloc(void ** ptr, long long N_bytes, int loc);

/// Free the CUDA device  memory pointed by 'gpuptr' (+errorcheck)
void gpu_free(void* gpuptr);
void pin_free(void* gpuptr);
/// Generalized free in loc 
void vec_free(void ** ptr, int loc);

/// Print Free/Total CUDA device memory along with 'message' (+errorcheck)
void gpu_showMem(char* message);

/// Initialize random values in vec of length using seed
void Dvec_init_cuRAND(double * dev_ptr, long long length, int seed);
void Svec_init_cuRAND(float * dev_ptr, long long length, int seed);

#endif
