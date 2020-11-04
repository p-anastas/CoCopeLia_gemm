///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some convinient C/C++ utilities for CoCopeLia.
///

#include "cpu_utils.hpp"

#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <omp.h>
#include <math.h>

timer_p init_timer(){
	timer_p timer = (timer_p) malloc(1*sizeof(struct timer_ctrl));
   	timer->alloc_t = 0; timer->h2d_t = 0; timer->d2h_t = 0; timer->cpu_ex_t = 0;
        timer->gpu_ex_t = 0; timer->buffer_t = 0; timer->reduce_t = 0; timer->total_t = 0;
	timer->A_chunk_send = 0; timer->B_chunk_send = 0; timer->C_chunk_send = 0;
	timer->C_chunk_get = 0;
	return timer; 
}

void report_timer(timer_p timer, char* timer_name){
    fprintf(stderr, "%s :\n"
	"Alloc time: \t%lf ms\n"
	"CPU exec time: \t%lf ms\n"
	"H2D time: \t%lf ms ( A_chunk_send = %lf ms, B_chunk_send = %lf ms, C_chunk_send = %lf ms)\n"
	"GPU exec time: \t%lf ms\n"
	"D2H time: \t%lf ms ( C_chunk_get = %lf ms)\n"
	"Total time: \t%lf ms\n\n",
	timer_name, 
	1000.0* timer->alloc_t,
	1000.0* timer->cpu_ex_t,
	1000.0* timer->h2d_t, 
	1000.0* timer->A_chunk_send, 
	1000.0* timer->B_chunk_send, 
	1000.0* timer->C_chunk_send, 
	1000.0* timer->gpu_ex_t,
	1000.0* timer->C_chunk_get, 
	1000.0* timer->d2h_t,
	1000.0* timer->total_t); 

}


long long dgemv_flops(size_t M, size_t N){
	return (long long) M * (2 * N + 1);
}

long long dgemv_bytes(size_t M, size_t N){
	return (M * N + N + M * 2)*sizeof(double) ; 
}

long long dgemm_flops(size_t M, size_t N, size_t K){
	return (long long) M * K * (2 * N + 1);
}

long long dgemm_bytes(size_t M, size_t N, size_t K){
	return (M * K + K * N + M * N * 2)*sizeof(double) ; 
}

long long sgemm_bytes(size_t M, size_t N, size_t K){
	return (M * K + K * N + M * N * 2)*sizeof(float) ; 
}

double Gval_per_s(long long value, double time){
  return value / (time * 1e9);
}

void massert(bool condi, const char* msg) {
  if (!condi) {
    fprintf(stderr, "Error: %s\n", msg);
    exit(1);
  }
}

void warning(const char* string) { fprintf(stderr, "WARNING ( %s )\n", string); }

void error(const char* string) {
  fprintf(stderr, "ERROR ( %s ) halting execution\n", string);
  exit(1);
}


double csecond(void) {
  struct timespec tms;

  if (clock_gettime(CLOCK_REALTIME, &tms)) {
    return (0.0);
  }
  /// seconds, multiplied with 1 million
  int64_t micros = tms.tv_sec * 1000000;
  /// Add full microseconds
  micros += tms.tv_nsec / 1000;
  /// round up if necessary
  if (tms.tv_nsec % 1000 >= 500) {
    ++micros;
  }
  return ((double)micros / 1000000.0);
}

double dabs(double x){
	if (x < 0) return -x;
	else return x;
}



double Drandom() {
  return (double)rand() / (double)RAND_MAX;
}

double Derror(double a, double b) {
  if (a == 0) return dabs(a - b); 
  return dabs(a - b)/a;
}


int Dequals(double a, double b, double eps) {
  double absA = dabs(a);
  double absB = dabs(b);
  double diff = dabs(a - b);

  if (a == b) {  // shortcut, handles infinities
    return 1;
  } else if (a == 0 || b == 0 || (absA + absB < DBL_MIN)) {
    // a or b is zero or both are extremely close to it
    // relative error is less meaningful here
    if (diff < (eps * DBL_MIN))
      return 1;
    else
      return 0;
  } else {  // use relative error
    if (diff /* /std::min((absA + absB), DBL_MIN)*/ < eps)
      return 1;
    else
      return 0;
  }
}

int Sequals(float a, float b, float eps) {
  float absA = fabs(a);
  float absB = fabs(b);
  float diff = fabs(a - b);

  if (a == b) {  // shortcut, handles infinities
    return 1;
  } else if (a == 0 || b == 0 || (absA + absB < FLT_MIN)) {
    // a or b is zero or both are extremely close to it
    // relative error is less meaningful here
    if (diff < (eps * FLT_MIN))
      return 1;
    else
      return 0;
  } else {  // use relative error
    if (diff /* /std::min((absA + absB), DBL_MIN)*/ < eps)
      return 1;
    else
      return 0;
  }
}

size_t Dvec_diff(double* a, double* b, long long size, double eps) {
	size_t failed = 0;
	#pragma omp parallel for
	for (long long i = 0; i < size; i++)
		if (!Dequals(a[i], b[i], eps)){
			#pragma omp atomic 
			failed++;
		}
	return failed;
}

size_t Svec_diff(float* a, float* b, long long size, float eps) {
  	size_t failed = 0;
	#pragma omp parallel for
  	for (long long i = 0; i < size; i++)
    		if (!Sequals(a[i], b[i], eps)){
			#pragma omp atomic 
			failed++;
		}
  	return failed;
}

short Stest_equality(float* C_comp, float* C, long long size) {
  size_t acc = 0, failed;
  float eps = 0.1;
  failed = Svec_diff(C_comp, C, size, eps);
  while (eps > FLT_MIN && !failed) {
    eps *= 0.1;
    acc++;
    failed = Svec_diff(C_comp, C, size, eps);
  }
  if (!acc) {
    fprintf(stderr, "Test failed %zu times\n", failed);
  } else
    fprintf(stderr, "Test passed(Accuracy= %zu digits, %zu/%lld breaking for %zu)\n\n",
            acc, failed, size, acc + 1);
  for (int i = 0; i < 10; i++)
    if (!Sequals(C_comp[i], C[i], eps))
      fprintf(stderr, "CPU vs GPU: %.15lf vs %.15lf\n", C_comp[i], C[i]);
  return (short) acc; 
}

short Dtest_equality(double* C_comp, double* C, long long size) {
  size_t acc = 0, failed;
  double eps = 0.1;
  failed = Dvec_diff(C_comp, C, size, eps);
  while (eps > DBL_MIN && !failed) {
    eps *= 0.1;
    acc++;
    failed = Dvec_diff(C_comp, C, size, eps);
  }
  if (!acc) {
    fprintf(stderr, "Test failed %zu times\n", failed);
  } else
    fprintf(stderr, "Test passed(Accuracy= %zu digits, %zu/%lld breaking for %zu)\n\n",
            acc, failed, size, acc + 1);
  for (int i = 0; i < 10; i++)
    if (!Dequals(C_comp[i], C[i], eps))
      fprintf(stderr, "Baseline vs Tested: %.15lf vs %.15lf\n", C_comp[i], C[i]);
  return (short) acc; 
}

void Dvec_init_hostRand(double* vec, long long size, int seed) {
 	if (!vec) error("Dvec_init_host-> vec is unallocated");
	srand(seed);
	//#pragma omp parallel for static
    	for (long long i = 0; i < size; i++) vec[i] = Drandom();
  return;
}

void Svec_init_hostRand(float* vec, long long size, int seed) {
	if (!vec) error("Svec_init_host-> vec is unallocated");
	srand(seed);
	//#pragma omp parallel for static
	for (long long i = 0; i < size; i++) vec[i] = (float) Drandom();
	return;
}


void optimized_memcpy(void* dest, void* src, long long bytes){
	memcpy(dest, src, bytes);
	return;

/*

	//long long div_bytes = bytes/8000, mod_bytes = bytes%8000;
	//#pragma omp parallel for
	//for (int i = 0; i < div_bytes; i++) memcpy(dest+i*8000, src + i*8000, 8000);
	//memcpy(dest+bytes-mod_bytes, src + bytes-mod_bytes, mod_bytes);
	//return;

	if (bytes < 2048){
		memcpy(dest, src, bytes);
		return;
	}
	int register threads;
	#pragma omp parallel
	{
		threads = omp_get_num_threads();
	}

	register long long thread_bytes = bytes/threads, mod1_bytes = bytes%threads;
	#pragma omp parallel for
	for (int i = 0; i < threads; i++) memcpy(dest+i*thread_bytes, src + i*thread_bytes, thread_bytes);
	memcpy(dest+threads*thread_bytes, src + threads*thread_bytes, mod1_bytes);

	return; 

	register int diviner = 80000;
	long long register byte_chunks, chunkparts, chunk_serial_rem, last_chunk, last_chunkparts, last_chunk_rem;	
	byte_chunks = bytes/diviner;
	chunkparts = diviner/threads;
	chunk_serial_rem = diviner%threads;
	last_chunk =  bytes%diviner;
	last_chunkparts = last_chunk/threads;
	last_chunk_rem = last_chunk%threads;

        for (int c = 0; c < byte_chunks; c++){
		#pragma omp parallel for
		for (int i = 0; i < threads; i++) memcpy(dest+c*diviner + i*chunkparts, src +c*diviner +i*chunkparts, chunkparts);
		memcpy(dest+c*diviner + chunkparts*threads , src +c*diviner + chunkparts*threads, chunk_serial_rem);
	}
	
	register long long offset = byte_chunks*diviner; 
	#pragma omp parallel for
	for (int i = 0; i < threads; i++) memcpy(dest+offset + i*last_chunkparts, src + offset + i*last_chunkparts, last_chunkparts);
	memcpy(dest+offset + last_chunkparts*threads, src + offset + last_chunkparts*threads, last_chunk_rem);
*/

}

void flush_cache(){
     const int size = 20*1024*1024; // Allocate 20M. Set much larger then L2
     char *c = (char *)malloc(size);
     for (int i = 0; i < 0xffff; i++)
       for (int j = 0; j < size; j++)
         c[j] = i*j;
}




