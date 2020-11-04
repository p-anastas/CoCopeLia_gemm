import pandas as pd

def GigaVal_per_s(val, time):
	return val * 1e-9 / time

def dgemm_flops(M,N,K):
	return M*N*(2*K+1)

machine_names= ['testbed-II_Tesla-V100']
funcs=['Dgemm']
version='1.1'
for func in funcs:
	for machine_num in range(len(machine_names)):
		machine = machine_names[machine_num]
		if func == 'Daxpy': # Not updated
			validata = pd.read_csv('../Results/%s/evaluation/%s_compare_libs_0.log' % (machine, func),
								   header=None, usecols=[0, 1, 2,  4, 5, 6],
								   names=['N','xloc', 'yloc', 'cublasXt_t', 'BLASX_t', 'streamed_t'])
		else:
			eval_CoCo = pd.read_csv('../Results/%s/evaluation/CoCopeLia_%s_0_v%s.log' % (machine, func, version),
							   header =None, usecols = [0,1,2,3,4,5,6,10], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc',  'CoCopeLia_t'])
			eval_BLASX = pd.read_csv('../Results/%s/evaluation/BLASX_%s_0.log' % (machine, func),
							   header =None, usecols = [0,1,2,3,4,5,6,10], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc',  'BLASX_t'])
			eval_cuBLASXt = pd.read_csv('../Results/%s/evaluation/cuBLASXt_%s_0.log' % (machine, func),
							   header =None, usecols = [0,1,2,3,4,5,6,10], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'cublasXt_t'])

			validata_0 = pd.merge(eval_CoCo, eval_BLASX, on = ['M', 'N', 'K', 'Aloc', 'Bloc', 'Cloc'])
			validata = pd.merge(validata_0, eval_cuBLASXt, on = ['M', 'N', 'K', 'Aloc', 'Bloc', 'Cloc'])

		print(validata.head(1))
		#print(validata['streamed_t'].max())
		if func == 'Daxpy':
			validata['size'] = validata['N']
			full_offload_samples = validata[(validata['size'] >= 8388608) &
											((validata['xloc'] == 1) & (validata['yloc'] == 1))]
			partial_offload_samples = validata[(validata['size'] >= 8 * 4096 ** 3) &
											   ((validata['xloc'] == 1) | (validata['yloc'] == 1))]
		else:
			validata['size'] = validata['M']*validata['N']*validata['K']
			full_offload_samples = validata[(validata['size'] >= 4096 ** 3) &
									#(validata['M'] >= validata['K']) &
									 ( (validata['Aloc'] == 1) & (validata['Bloc'] == 1) & (validata['Cloc'] == 1)) ]
			partial_offload_samples = validata[(validata['size'] >= 4096 ** 3) &
											 #(validata['M'] >= validata['K']) &
											 #  ((validata['Aloc'] != 0) | (validata['Bloc'] != 0) | (validata['Cloc'] != 0)) &
											((validata['Aloc'] == 1) | (validata['Bloc'] == 1) | (validata['Cloc'] == 1))]
			#validata_samples['flops'] = dgemm_flops(validata_samples['M'],validata_samples['N'],validata_samples['K'])
			#full_offload_CoCoBLAS_speedup = (full_offload_samples['cublasXt_t'] - full_offload_samples['CoCopeLia_t'])/\
			#								full_offload_samples['cublasXt_t']
			#full_offload_BLASX_speedup = (full_offload_samples['cublasXt_t'] - full_offload_samples['BLASX_t'])/\
			#							 full_offload_samples['cublasXt_t']
			#partial_offload_CoCoBLAS_speedup = (partial_offload_samples['cublasXt_t'] - partial_offload_samples['CoCopeLia_t']) / \
			#								partial_offload_samples['cublasXt_t']
			#partial_offload_BLASX_speedup = (partial_offload_samples['cublasXt_t'] - partial_offload_samples['BLASX_t']) / \
			#							 partial_offload_samples['cublasXt_t']

		print('Function: %s Machine : %s' % (func, machine))
		#print(validata_plot)

		#print("Full offload ({A,B,C}loc = 1) Speedup CoCoBLAS: %lf " % (100*full_offload_CoCoBLAS_speedup.mean()))
		#print("Full offload ({A,B,C}loc = 1) Speedup BLASX: %lf " % (100 * full_offload_BLASX_speedup.mean()))
		#print("Partial offload Speedup CoCoBLAS: %lf " % (100*partial_offload_CoCoBLAS_speedup.mean()))
		#print("Partial offload Speedup BLASX: %lf " % (100 * partial_offload_BLASX_speedup.mean()))

		# SOTA = State of the art
		full_offload_samples['SOTA'] = full_offload_samples[['cublasXt_t', 'BLASX_t']].min(axis=1)
		full_offload_CoCoBLAS_speedup = (full_offload_samples['SOTA'] - full_offload_samples['CoCopeLia_t'])/\
										full_offload_samples['SOTA']

		partial_offload_samples['SOTA'] = partial_offload_samples[['cublasXt_t', 'BLASX_t']].min(axis=1)
		partial_offload_CoCoBLAS_speedup = (partial_offload_samples['SOTA'] - partial_offload_samples['CoCopeLia_t']) / \
										partial_offload_samples['SOTA']

		print("Full offload ({A,B,C}loc = 1) Speedup CoCoBLAS: %lf " % (100*full_offload_CoCoBLAS_speedup.mean()))
		print("Partial offload Speedup CoCoBLAS: %lf " % (100*partial_offload_CoCoBLAS_speedup.mean()))
