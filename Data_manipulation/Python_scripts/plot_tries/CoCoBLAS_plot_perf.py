import pandas as pd
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from pylab import cm

def GigaVal_per_s(val, time):
	return val * 1e-9 / time

def dgemm_flops(M,N,K):
	return M*N*(2*K+1)

font=8
plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=font)
plt.rc('ytick', labelsize=font)
plt.rc('axes', labelsize=font)

# width as measured in inkscape
width = 3.487
height = width / 1.618

# Generate colors from the 'viridis' colormap
colors = cm.get_cmap('viridis',  4)
colors1 = cm.get_cmap('magma',  4)
color_mine = colors(0)
color_cublasxt = colors(2)
color_ideal = colors(3)
color_blasx = colors1(2)


machine_names= ['testbed-II_Tesla-V100']
funcs=['Dgemm']
version='1.1'
machine_flops = [(1,8.5)] #(0.5, 1.35), 
for func in funcs:
	for machine_num in range(len(machine_names)):
		machine = machine_names[machine_num]
		machine_FLOP = machine_flops[machine_num]

		eval_CoCo = pd.read_csv('../Results/%s/evaluation/CoCopeLia_%s_0_v%s.log' % (machine, func, version),
							   header = None, usecols = [0,1,2,3,4,5,6,9], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc',  'CoCopeLia_t'])
		eval_BLASX = pd.read_csv('../Results/%s/evaluation/BLASX_%s_0.log' % (machine, func),
							   header = None, usecols = [0,1,2,3,4,5,6,9], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc',  'BLASX_t'])
		eval_cuBLASXt = pd.read_csv('../Results/%s/evaluation/cuBLASXt_%s_0.log' % (machine, func),
							   header = None, usecols = [0,1,2,3,4,5,6,9], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'cublasXt_t'])

		validata_0 = pd.merge(eval_CoCo, eval_BLASX, on = ['M', 'N', 'K', 'Aloc', 'Bloc', 'Cloc'])
		validata = pd.merge(validata_0, eval_cuBLASXt, on = ['M', 'N', 'K', 'Aloc', 'Bloc', 'Cloc'])
		
		#print(validata.head(1))
		#print(validata['CoCopeLia_t'].max())

		validata['size'] = validata['M']*validata['N']*validata['K']
		validata_plot = validata[(validata['size'] >= 4096**3) & 
								 #(validata['K'] > validata['M']) & (validata['K'] > validata['N']) & # Thin-by-fat
								 (validata['K'] == validata['M']) & (validata['K'] == validata['N']) & # Square
								 ( (validata['Aloc'] == 1) & (validata['Bloc'] == 1) & (validata['Cloc'] == 1)) ] # Full offload
								 #( (validata['Aloc'] == 1) | (validata['Bloc'] == 1) | (validata['Cloc'] == 1)) ] # Partial offload
		#validata_plot['size'] = validata_plot['M']*validata_plot['N']*validata_plot['K']
		streamed_list = list(validata_plot['CoCopeLia_t'])
		cublasXt_list = list(validata_plot['cublasXt_t'])
		BLASX_list = list(validata_plot['BLASX_t'])
		flop_list = dgemm_flops(validata_plot['M'],
								validata_plot['N'],validata_plot['K'])
		xaxis_list = list(validata_plot['size'])

		#print(validata_plot)
		print(xaxis_list)
		print(streamed_list)
		print(cublasXt_list)

		fig, ax = plt.subplots()
		# fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
		fig.subplots_adjust(left=.15, bottom=.17, right=.99, top=.97)

		plt.plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, cublasXt_list)), 's', markersize=2, alpha=0.8, color=color_cublasxt, label='cuBLASXt')
		plt.plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, BLASX_list)), '^', markersize=2, alpha=0.8, color=color_blasx, label='BLASX')
		plt.plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, streamed_list)), 'o', markersize=2, alpha=1, color=color_mine, label='CoCoPeLia')
		plt.legend(fontsize=font, loc='upper center', fancybox = False, ncol=3)

		ax.set_ylabel('Performance (Tflops/s)')
		ax.set_xlabel('Problem size (MxNxK)')

		ax.set_ylim(machine_FLOP)

		fig.set_size_inches(width, height)
		fig.savefig('CoCoPeLia_perf_full_sq_%s.pdf' %machine)
		#fig.savefig('CoCoPeLia_perf_%s.png' %machine)
		plt.close()

