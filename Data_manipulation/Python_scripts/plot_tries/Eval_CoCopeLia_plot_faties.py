import pandas as pd
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from pylab import cm
import numpy as np

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
colors_p = cm.get_cmap('viridis',  3)
colors_p_1 = cm.get_cmap('magma',  5)


machine_names= ['testbed-II_Tesla-V100']
machine_plot_lims = [(0,8.5), (0,17)] #(0.5, 1.35), 
funcs=['Dgemm']
version='1.2'
loc=[1,1,1]
colors=[colors_p(0),colors_p(1),colors_p(2),colors_p(3),colors_p(4)]
mid_sizes=[12000]
#symbolist=['-s','--^','-.o']
ctr = 0
for func in funcs:
	for machine_num in range(len(machine_names)):
		machine = machine_names[machine_num]

		eval_CoCo = pd.read_csv('../Results/%s/evaluation/CoCopeLia_%s_0_v%s.log' % (machine, func, version), # 8 = avg, 9 = min, 10 = max, 11 = first
							   header = None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T_CoCo', 'Aloc','Bloc', 'Cloc',  'CoCopeLia_t'])
		eval_BLASX = pd.read_csv('../Results/%s/evaluation/BLASX_%s_0.log' % (machine, func), # 8 = avg, 9 = min, 10 = max
							   header = None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T_BLASX', 'Aloc','Bloc', 'Cloc',  'BLASX_t'])
		eval_cuBLASXt = pd.read_csv('../Results/%s/evaluation/cuBLASXt_%s_0.log' % (machine, func), # 8 = avg, 9 = min, 10 = max, 11 = first
							   header = None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T_cuBLAS', 'Aloc','Bloc', 'Cloc', 'cublasXt_t'])

		eval_Serial = pd.read_csv('../Results/%s/evaluation/Serial_%s_0.log' % (machine, func), # 7 = avg, 8 = min, 9 = max, 10 = first, 11 = h2d, 12 = d2h
							   header = None, usecols = [0,1,2,3,4,5,7,11,12], names= ['M','N','K', 'Aloc','Bloc', 'Cloc', 'Serial_t', 'h2d_t', 'd2h_t'])

		print( "Read %d values from \"../Results/%s/evaluation/CoCopeLia_%s_0_v%s.log\"..." %( len(eval_CoCo), machine, func, version))
		print( "Read %d values from \"../Results/%s/evaluation/BLASX_%s_0.log\".." %( len(eval_BLASX), machine, func))
		print( "Read %d values from \"../Results/%s/evaluation/cuBLASXt_%s_0.log\".." %( len(eval_cuBLASXt), machine, func))
		print( "Read %d values from \"../Results/%s/evaluation/Serial_%s_0.log\".." %( len(eval_Serial), machine, func))

		validata_0 = pd.merge(eval_CoCo, eval_BLASX, on = ['M', 'N', 'K', 'Aloc', 'Bloc', 'Cloc'])
		validata_01 = pd.merge(validata_0, eval_cuBLASXt, on = ['M', 'N', 'K', 'Aloc', 'Bloc', 'Cloc'])
		validata = pd.merge(validata_01, eval_Serial, on = ['M', 'N', 'K', 'Aloc', 'Bloc', 'Cloc'])
		print( "Combined in dataframe of length = %d with head:" %( len(validata)))
		print(validata.head(1))
		#print(validata['CoCopeLia_t'].max())
		#

		fig, ax = plt.subplots()
		#fig.subplots_adjust(left=.15, bottom=.17, right=.99, top=.97)
		#fig.set_size_inches(width, height)

		validata_filtered = validata[((validata['Aloc'] == loc[0]) & (validata['Bloc'] == loc[1]) & (validata['Cloc'] == loc[2]))]
		print( "Kept %d Fat & thin values" %( len(validata_filtered)))

		for size in mid_sizes:
			validata_plot = validata_filtered[(validata['K']*validata['M']*validata['N'] <= (size**3)) & ((validata['K']+1)*(validata['M']+1)*(validata['N']+1) >= (size-1)**3)]# &# Problem set
					#(validata['M']!= 1600) & (validata['M']!= 1714) & (validata['M']!= 2000) & 
					#(validata['M']!= 2666) & (validata['M']!= 3000) & (validata['M']!= 4000) & 
					#(validata['M']!= 6000) & (validata['M']!= 8000) & (validata['M']!= 18000)] # Custom removal of extras
			print(validata_plot)	
			print( "Plotting line for mid_size=%d with %d points..." %( size, len(validata_plot)))

			mem_foot = list(validata_plot['M']*validata_plot['N'] + validata_plot['K']*validata_plot['N'] + validata_plot['M']*validata_plot['K'])
			if func == 'Dgemm' :
				mem_foot = list(map(lambda x: float(int(8*x/1024/1024/1024*100)/100), mem_foot))
			elif func == 'Sgemm' :
				mem_foot = list(map(lambda x: float(int(4*x/1024/1024/1024*100)/100), mem_foot))

			streamed_list = list(validata_plot['CoCopeLia_t'])
			cublasXt_list = list(validata_plot['cublasXt_t'])
			BLASX_list = list(validata_plot['BLASX_t'])
			Serial_list = list(validata_plot['Serial_t'])

			exec_list = list(validata_plot['Serial_t']- validata_plot['h2d_t'] - validata_plot['d2h_t'])
			h2d_list = list(validata_plot['h2d_t'])
			d2h_list = list(validata_plot['d2h_t'])
			Optimus_list = map(lambda x,y,z: max(max(x,y),z), exec_list, h2d_list, d2h_list)

			flop_list = dgemm_flops(validata_plot['M'], validata_plot['N'],validata_plot['K'])
			xaxis_list = list(validata_plot['M'])
			_, streamed_list = zip(*sorted(zip(xaxis_list, streamed_list), key=lambda t: t[0]))
			_, cublasXt_list = zip(*sorted(zip(xaxis_list, cublasXt_list), key=lambda t: t[0]))
			_, Serial_list = zip(*sorted(zip(xaxis_list, Serial_list), key=lambda t: t[0]))
			_, Optimus_list = zip(*sorted(zip(xaxis_list, Optimus_list), key=lambda t: t[0]))
			_, mem_foot = zip(*sorted(zip(xaxis_list, mem_foot), key=lambda t: t[0]))
			xaxis_list, BLASX_list = zip(*sorted(zip(xaxis_list, BLASX_list), key=lambda t: t[0]))	

			print(list(mem_foot))
			print(xaxis_list)

			flag = -1
			mem_foot_normalized = []
			for elem in list(mem_foot):
				if elem == min(mem_foot):
					flag = 1
				mem_foot_normalized.append(flag*(elem-min(mem_foot)))
			print(mem_foot_normalized)
			x = mem_foot_normalized

			plt.plot(x, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, streamed_list)), 
			'-s', alpha=1, linewidth = 3, markersize = 5, color=colors[0], label='CoCoPeLia' )

			plt.plot(x, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, cublasXt_list)), 
			'-^', alpha=1, markersize = 5, color=colors[1], label='cuBLASXt' )

			plt.plot(x, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, BLASX_list)), 
			'-o', alpha=1, markersize = 5, color=colors[2], label='BLASX' )


			#plt.plot(x, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, Serial_list)), 
			#'-', alpha=1, markersize = 5, color=colors[3], label='Serialized Transfers' )

			plt.plot(x, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, Optimus_list)), 
			'--', alpha=1, linewidth = 1, color='r')


			plt.axvline(x=0,linewidth=1, linestyle = '--', color='k')
			plt.text(-2, 7.2, 'M = N = K', fontsize=font, color='k')
			plt.text(-8.5, 4, ' Fat-by-thin (M,N < K)  <<', fontsize=font, rotation = 60, color='k')
			plt.text(3, 4, '>>  Thin-by-fat (M,N > K)', fontsize=font, rotation = -60, color='k')

			plt.text(10.5, 3, 'Full overlap', fontsize=font, rotation = 0, color='r') # 0.42

			if func == 'Dgemm' :
				ax.set_xticklabels([0,17,10,5,3,5,10,17])
			elif func == 'Sgemm' :
				ax.set_xticklabels([0,8.5,5,2.5,1.5,2.5,5,8.5])
			handles, labels = ax.get_legend_handles_labels()
			# sort both labels and handles by labels
			# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
			ax.legend(handles, labels, fontsize=font, loc='upper center', fancybox = False, ncol=3)

			ax.set_title("Static Comp. Complexity M*N*K=%d^3" %(size))
			
			ax.set_ylabel('Performance (Tflops/s)')
			ax.set_xlabel('Memory footprint(GB)')

			ax.set_ylim(machine_plot_lims[ctr*len(machine_names) +machine_num])

			fig.savefig('CoCoPeLia_%s_eval_faties_%d_%s.pdf' % (func, size, machine))
			plt.close()
	ctr +=1
