import pandas as pd
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from pylab import cm
import numpy as np
from collections import OrderedDict

import input_parsing as parse
import plot_stuff as mplt 

def GigaVal_per_s(val, time):
	return val * 1e-9 / time

def dgemm_flops(M,N,K):
	return M*N*(2*K+1)

import seaborn

font=8
plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=font)
plt.rc('ytick', labelsize=font)
plt.rc('axes', labelsize=font)

# width as measured in inkscape
width = 3.487
height = width / 1.618*3/4


machine_names= ['testbed-I_Tesla-K40','testbed-II_Tesla-V100']
funcs=['Dgemm']#, 'Sgemm']
version='final'
exchept_T_list = [512, 768,8192+512,8192+1024, 8192+512+1024, 8192+2048, 8192+512 +2048 ]
for func in funcs:
	for machine in machine_names:

		validation_data = parse.read_validation_values("..",machine,func,version)	
		
		#Square set
		locs= [[1,1,1]]
		mid_sizes= [8192,16384]
		ctrs_fat = [2,3]
		ctrs_thin = []
		square_validation_set = parse.validation_set_split_BLAS3("Square",locs,mid_sizes,ctrs_fat,ctrs_thin,validation_data)
		square_validation_set = parse.cleanT(square_validation_set,exchept_T_list)
		#print(square_validation_set)

		fig, ax = plt.subplots()
		fig.subplots_adjust(left=.15, bottom=.17*4/3, right=.99, top=.97)

		teams = square_validation_set.groupby(['M', 'N' , 'K', 'Aloc', 'Bloc', 'Cloc'])
		n = 0
		maxT = []
		for state, curr_set in teams:
			maxT.append(int(curr_set.iloc[curr_set['Noreuse_t'].argmin()]['T']))
			#maxT.append(int(curr_set.iloc[curr_set['CoCopeLia_t'].argmin()]['T']))
			plt.axvline(x=maxT[n],linewidth=0.5, linestyle = '--', color=mplt.colors2_2(n))
			n += 1
		n = 0
		for state, curr_set in teams:
			plot_x, plot_y = zip(*sorted(zip(list(curr_set['T']), list(map(lambda x: GigaVal_per_s(dgemm_flops(curr_set['M'],curr_set['N'],curr_set['K']),x*1024), curr_set['Noreuse_t']))), key=lambda t: t[0]))
			plt.plot(plot_x, plot_y,
				 '-o', linewidth= 0.5, markersize=2, color=mplt.colors2_2(n), label='M,N=%dK, K = %dK' % (state[0]/1024, state[2]/1024))
			#plt.plot(curr_set['T'], list(map(lambda x: GigaVal_per_s(dgemm_flops(curr_set['M'],curr_set['N'],curr_set['K']),x*1024), curr_set['CoCopeLia_t'])),
			#	 '--^', linewidth= 0.5, markersize=3, color=mplt.colors2_2(n+1), label='D1,D2=%dK, D3 = %dK' % (state[0]/1024, state[2]/1024))

			n += 1

		y_bot = plt.gca().get_ylim()[0]
		for ctr in range(n):
			plt.text(maxT[ctr]+ 120, y_bot*1.1, str(maxT[ctr]), rotation=90, fontsize=font, color=mplt.colors2_2(ctr)) # 0.42

		handles, labels = fig.gca().get_legend_handles_labels()
		by_label = OrderedDict(zip(labels, handles))

		ax.set_ylabel('Performance (Tflops/s)')
		#plt.yscale('log')
		ax.set_xlabel('Tiling size $T$')

		#ax.set_ylim(1,10)

		fig.set_size_inches(width, height)
		if machine=='testbed-I_Tesla-K40':
			fig.subplots_adjust(left=.15, bottom=.18, right=.99, top=.80)
			fig.legend(by_label.values(), by_label.keys(),
				loc="upper center",   # Position of legend
				borderaxespad=0.1,    # Small spacing around legend box
				#title="Model",  # Title for the legend
				fontsize=font, fancybox = False, ncol=2
				)
			fig.set_size_inches(width, height*1.25)
		fig.savefig('Plots/motivation_tile_selection_sys-%s_func-%s.pdf' % (machine, func))
		plt.close()

