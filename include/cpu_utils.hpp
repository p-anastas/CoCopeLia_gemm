///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#include <stdio.h>
#include <cstring>

#define ITER 100

typedef struct timer_ctrl {
  double alloc_t, h2d_t, d2h_t, cpu_ex_t,
         gpu_ex_t, buffer_t, reduce_t, total_t, 	
	 A_chunk_send, B_chunk_send, C_chunk_send, C_chunk_get;
} * timer_p;

timer_p init_timer();
	
void report_timer(timer_p timer, char* timer_name);

long long dgemv_flops(size_t M, size_t N);
long long dgemv_bytes(size_t M, size_t N);
long long dgemm_flops(size_t M, size_t N, size_t K); 
long long dgemm_bytes(size_t M, size_t N, size_t K); 
long long sgemm_bytes(size_t M, size_t N, size_t K);

double Gval_per_s(long long value, double time);

void massert(bool condi, const char* msg);
void error(const char* string);
void warning(const char* string);

double csecond();

double dabs(double x);

double deviation(double y_pred, double y_value);

double Drandom(double min, double max);
size_t Dvec_diff(double* a, double* b, size_t size, double eps);
double Derror(double a, double b);
int Dequals(double a, double b, double eps);
short Dtest_equality(double* C_comp, double* C, long long size);
short Stest_equality(float* C_comp, float* C, long long size);

void Dvec_init_hostRand(double* vec, long long size, int seed);
void Svec_init_hostRand(float* vec, long long size, int seed) ;

void optimized_memcpy(void* dest, void* src, long long bytes); 

void flush_cache();
#endif
