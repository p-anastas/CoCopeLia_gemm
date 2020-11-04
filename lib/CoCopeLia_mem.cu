///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some helper functions related to allocation and data initialization. 
///

#include <CoCopeLia.hpp>
#include <gpu_utils.hpp>
#include <cpu_utils.hpp>

void * CoComalloc(long long bytes, short loc, short dev){
	int count; 
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoComalloc: cudaGetDeviceCount failed");
	massert(dev >=0 && dev < count, "CoComalloc: Invalid input device number");
	/// Set device 
	cudaSetDevice(dev);

	void * out_ptr; 
	if (loc == 1) out_ptr = pin_malloc(bytes); //malloc(bytes);// 
	else if (loc == 0) out_ptr = gpu_malloc(bytes);
	else error("CoComalloc : invalid loc value");
	return out_ptr; 
}

void Dvec_init_CoCoRand(double *vec, long long length, int seed, short loc, short dev){
	int count; 
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "Dvec_init_CoCo: cudaGetDeviceCount failed");
	massert(dev >=0 && dev < count, "Dvec_init_CoCo: Invalid input device number");
	/// Set device 
	cudaSetDevice(dev);

	if (loc == 1) Dvec_init_hostRand(vec, length, seed); 
	else if (loc == 0) Dvec_init_cuRAND(vec, length, seed);
	else error("Dvec_init_CoCo : invalid loc value");
}

void Svec_init_CoCoRand(float *vec, long long length, int seed, short loc, short dev){
	int count; 
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "Dvec_init_CoCo: cudaGetDeviceCount failed");
	massert(dev >=0 && dev < count, "Dvec_init_CoCo: Invalid input device number");
	/// Set device 
	cudaSetDevice(dev);

	if (loc == 1) Svec_init_hostRand(vec, length, seed); 
	else if (loc == 0) Svec_init_cuRAND(vec, length, seed);
	else error("Dvec_init_CoCo : invalid loc value");
}

void vec_get_memcpy(void* dest, void* src,  long long bytes, short dev){
	int count; 
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "vec_get_memcpy: cudaGetDeviceCount failed");
	massert(dev >=0 && dev < count, "vec_get_memcpy: Invalid input device number");
	/// Set device 
	cudaSetDevice(dev);
	massert(CUBLAS_STATUS_SUCCESS == cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToHost), "vec_get_memcpy: cudaMemcpy failed");
}

void vec_set_memcpy(void* dest, void* src,  long long bytes, short dev){
	int count; 
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "vec_set_memcpy: cudaGetDeviceCount failed");
	massert(dev >=0 && dev < count, "vec_set_memcpy: Invalid input device number");
	/// Set device 
	cudaSetDevice(dev);
	massert(CUBLAS_STATUS_SUCCESS == cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice), "vec_set_memcpy: cudaMemcpy failed");
}
