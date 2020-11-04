///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The model from "Performance models for CPU-GPU data transfers"
///
/// Werkhoven:
/// Overlap model for no implicit synchronization and 2 copy engines:
/// time = max(tThd + tE/nStreams + tTdh/nStreams, 
/// tThd/nStreams + tE + tTdh/nStreams,
/// tThd/nStreams + tE/nStreams + tTdh)
/// Single Data transfer: LogP
/// Large transfer with multiple streams LogGP
/// Total model for streaming overlap with nStreams blocks
/// time = max(L + o + B_sent*Gh2d + g*(nStreams − 1) + tE/nStreams + L + o + B_recv*Gd2h/nStreams + g*(nStreams − 1), 
/// L + o + B_recv*Gh2d/nStreams + g*(nStreams − 1) + tE + L + o + B_recv*Gd2h/nStreams + g*(nStreams − 1),
/// L + o + B_sent*Gh2d/nStreams + g*(nStreams − 1) + tE/nStreams + L + o + B_recv*Gd2h + g*(nStreams − 1))
/// TODO: We use for this model g = L + o since the actual g is very hard to calculate accurately (same as werkhoven). Since g <= L + o, this results in a small time overestimation, 
///	  in our case this is not a problem since Werkhoven always underestimates time because of its linearily.

#include <stdlib.h>
#include <math.h>

#include "CoCopeLia_CoModel.hpp"
#include "CoCopeLia_GPUexec.hpp"
#include "Werkhoven.hpp"
#include "cpu_utils.hpp"


/// Initializes underlying models required for Werkhoven
WerkhovenModel_p WerkhovenModel_init(short dev_id, char* func, short level, short mode)
{
	WerkhovenModel_p out_model = (WerkhovenModel_p) malloc(1*sizeof(struct werkhoven_model)); 
	out_model->h2d = CoModel_init(dev_id, -1);
	out_model->d2h = CoModel_init(-1, dev_id);
	//if(level == 1) out_model->GPUexec_model_ptr = (void*) GPUexec1Model_init(dev_id, func, mode);
	//else if(level == 2) out_model->GPUexec_model_ptr = (void*) GPUexec2Model_init(dev_id, func, mode);
	//else 
	if(level == 3) out_model->GPUexec_model_ptr = (void*) GPUexec3Model_init(dev_id, func, mode);
	return out_model;

}

///  Predicts 3-way overlaped execution time for nStream (equal) data blocks of any kernel using Werkhoven's model. 
double WerkhovenModel_predict(WerkhovenModel_p model, long long h2d_bytes, long long d2h_bytes, double nStreams, short level, size_t D1,  size_t D2, size_t D3)
{
	double serial_time, h2d_dom_time, ker_dom_time, d2h_dom_time, result, d2h_time[2], h2d_time[2], kernel_time[2];
	if (!nStreams) return 0.0;
	/// T[0] = Serialized/Dominant, T[1] = Blocked with nStreams 
	h2d_time[0] = model->h2d->ti + model->h2d->tb*h2d_bytes + model->h2d->ti*(nStreams - 1);
	h2d_time[1] = model->h2d->ti + model->h2d->tb*h2d_bytes/nStreams;

	// Linear performance assumption used by werkhoven. 
	double t_kernel = 0;
	//if(level == 1) t_kernel = GPUexec1Model_predict((GPUexec1Model_p)model->GPUexec_model_ptr, D1); 
	//else if(level == 2) t_kernel = GPUexec2Model_predict((GPUexec2Model_p)model->GPUexec_model_ptr, D1, D2); 
	//else 
	if(level == 3){
		size_t T = fmin(maxDim_blas3, fmin(fmin(D1,D2), D3));
		if ((T-minDim_blas3)%step_blas3) T = (T-minDim_blas3)/step_blas3*step_blas3;
		t_kernel = (D1*1.0/T * D2*1.0/T * D3*1.0/T)* GPUexec3Model_predict((GPUexec3Model_p)model->GPUexec_model_ptr, T); 
	}
	else error("WerkhovenModel_predict: invalid BLAS level");

	kernel_time[0] = t_kernel; 
	kernel_time[1] = t_kernel/nStreams;
	d2h_time[0] = model->d2h->ti + model->d2h->tb*d2h_bytes + model->d2h->ti*(nStreams - 1);
	d2h_time[1] = model->d2h->ti + model->d2h->tb*d2h_bytes/nStreams;

	h2d_dom_time = h2d_time[0] + kernel_time[1] + d2h_time[1]; 
	ker_dom_time = h2d_time[1] + kernel_time[0] + d2h_time[1];
	d2h_dom_time = h2d_time[1] + kernel_time[1] + d2h_time[0];
	serial_time = model->h2d->ti + model->h2d->tb*h2d_bytes + t_kernel +  model->d2h->ti + model->d2h->tb*d2h_bytes; 

	result =  fmax(fmax(h2d_dom_time, ker_dom_time), d2h_dom_time); 

	fprintf(stderr, "Werkhoven predicted ( Streams = %lf) -> ", nStreams); 
	double res_h2d = h2d_time[1], res_ker = kernel_time[1], res_d2h = d2h_time[1]; 
	if (h2d_dom_time == result) {
		res_h2d = h2d_time[0]; 
		fprintf(stderr, "H2D dominant problem\n");
	}
	if (ker_dom_time == result) {
		res_ker = kernel_time[0]; 
		fprintf(stderr, "Exec dominant problem\n");
	}
	if (d2h_dom_time == result){
		res_d2h = d2h_time[0];
		fprintf(stderr, "D2h dominant problem\n");
	}
	
	double overlap_speedup = ( serial_time - result) / serial_time; 
	
	fprintf(stderr, "\tt_h2d: %lf ms\n"
	"\tt_exec: %lf ms\n"
	"\tt_d2h: %lf ms\n"
	"\t -> total: %lf ms (%lf GFlops/s)\n"
	"\tExpected Speedup = \t%.3lf\n\n", 
	1000*res_h2d, 1000*res_ker, 1000*res_d2h, 1000*result, Gval_per_s(dgemm_flops(D1,D2,D3), result), overlap_speedup);
	
	return result; 
}
