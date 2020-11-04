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

# width as measured in inkscape
width = 3.487
height = width / 1.618

# Generate colors from the 'viridis' colormap
#colors = cm.get_cmap('viridis',  4)
#colors1 = cm.get_cmap('magma',  4)
#color_mine = colors(0)
#color_cublasxt = colors(2)
#color_ideal = colors(3)
#color_blasx = colors1(2)


machine_names= ['testbed-II_Tesla-V100', 'testbed-I_Tesla-K40'] #
machine_plot_lims = [(1,8.5),(0.4, 1.4), (2.8,17),  (0,4)] # 
funcs=['Dgemm','Sgemm']
version='1.2'
loc_list=[[1,1,1],[1,1,0],[0,0,1]] # 1. Full offload, 2. C available, 3. A,B available (unmodified)
colors=['r','g','b']
#symbolist=['-s','--^','-.o']
ctr = 0
for func in funcs:
	for machine_num in range(len(machine_names)):
		plt.rc('font', family='serif', serif='Times')
		#plt.rc('text', usetex=True)
		plt.rc('xtick', labelsize=font)
		plt.rc('ytick', labelsize=font)
		plt.rc('axes', labelsize=font)

		machine = machine_names[machine_num]

		eval_CoCo = pd.read_csv('../Results/%s/evaluation/CoCopeLia_%s_0_v%s.log' % (machine, func, version),
							   header = None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T_CoCo', 'Aloc','Bloc', 'Cloc',  'CoCopeLia_t'])
		eval_BLASX = pd.read_csv('../Results/%s/evaluation/BLASX_%s_0.log' % (machine, func),
							   header = None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T_BLASX', 'Aloc','Bloc', 'Cloc',  'BLASX_t'])
		eval_cuBLASXt = pd.read_csv('../Results/%s/evaluation/cuBLASXt_%s_0.log' % (machine, func),
							   header = None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T_cuBLAS', 'Aloc','Bloc', 'Cloc', 'cublasXt_t'])

		print( "Read %d values from \"../Results/%s/evaluation/CoCopeLia_%s_0_v%s.log\"..." %( len(eval_CoCo), machine, func, version))
		print( "Read %d values from \"../Results/%s/evaluation/BLASX_%s_0.log\".." %( len(eval_BLASX), machine, func))
		print( "Read %d values from \"../Results/%s/evaluation/cuBLASXt_%s_0.log\".." %( len(eval_cuBLASXt), machine, func))

		validata_0 = pd.merge(eval_CoCo, eval_BLASX, on = ['M', 'N', 'K', 'Aloc', 'Bloc', 'Cloc'])
		validata = pd.merge(validata_0, eval_cuBLASXt, on = ['M', 'N', 'K', 'Aloc', 'Bloc', 'Cloc'])
		print( "Combined in dataframe of length = %d with head:" %( len(validata)))
		print(validata.head(1))
		#print(validata['CoCopeLia_t'].max())
		#

		fig, ax = plt.subplots()
		#fig.subplots_adjust(left=.15, bottom=.17, right=.99, top=.97)
		fig.set_size_inches(width, height)

		validata_filtered = validata[(validata['K'] == validata['M']) & (validata['K'] == validata['N'])] # Square
		print( "Kept %d square values" %( len(validata_filtered)))
		for loc_num in range(len(loc_list)):
			loc = loc_list[loc_num]
			validata_plot = validata_filtered[((validata['Aloc'] == loc[0]) & (validata['Bloc'] == loc[1]) & (validata['Cloc'] == loc[2]))]	
			print( "Plotting line for loc=[%d,%d,%d] with %d points..." %( loc[0], loc[1], loc[2], len(validata_plot)))
			streamed_list = list(validata_plot['CoCopeLia_t'])
			cublasXt_list = list(validata_plot['cublasXt_t'])
			BLASX_list = list(validata_plot['BLASX_t'])
			flop_list = dgemm_flops(validata_plot['M'], validata_plot['N'],validata_plot['K'])
			xaxis_list = list(validata_plot['M'])

			_, streamed_list = zip(*sorted(zip(xaxis_list, streamed_list), key=lambda t: t[0]))
			_, cublasXt_list = zip(*sorted(zip(xaxis_list, cublasXt_list), key=lambda t: t[0]))
			_, BLASX_list = zip(*sorted(zip(xaxis_list, BLASX_list), key=lambda t: t[0]))
			xaxis_list, flop_list = zip(*sorted(zip(xaxis_list, flop_list), key=lambda t: t[0]))	

			plt.plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, cublasXt_list)), 
			'-o', alpha=1, markersize = 1, color=colors[loc_num], label='cuBLASXt loc-[%d,%d,%d]' % (loc[0], loc[1], loc[2]) )

			plt.plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, BLASX_list)), 
			'--^', alpha=1, markersize = 1, color=colors[loc_num], label='BLASX loc-[%d,%d,%d]' % (loc[0], loc[1], loc[2]))

			plt.plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, streamed_list)), 
			'-', alpha=1, linewidth = 1, color=colors[loc_num], label='CoCoPeLia loc-[%d,%d,%d]' % (loc[0], loc[1], loc[2]))
			
		handles, labels = ax.get_legend_handles_labels()
		# sort both labels and handles by labels
		labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
		ax.legend(handles, labels, fontsize=font, loc='upper center', fancybox = False, ncol=3)

		ax.set_title("%s@%s" %( func, machine))
		ax.set_ylabel('Performance (Tflops/s)')
		ax.set_xlabel('Problem size (N*N)N')

		ax.set_ylim(machine_plot_lims[ctr*len(machine_names) +machine_num])

		fig.savefig('CoCoPeLia_%s_eval_sq_%s.pdf' % (func, machine))
		plt.close()
	ctr +=1

