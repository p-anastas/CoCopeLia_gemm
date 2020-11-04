import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import math
import numpy as np
from collections import OrderedDict

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

import input_parsing 

machine_names= ['testbed-I_Tesla-K40', 'testbed-II_Tesla-V100'] #
cool_names= ['testbed-I', 'testbed-II'] 
funcs=['Dgemm','Sgemm'] # 
version='final'

font=8
plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=font-1)
plt.rc('ytick', labelsize=font-1)
plt.rc('axes', labelsize=font)


# width as measured in inkscape
width = 3.487
height = width / 1.618*5/4

fig, axs = plt.subplots( len(funcs), len(machine_names), sharey=True, sharex=True)
#fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
fig.subplots_adjust(left=.15, bottom=.23, right=.99, top=.90)

import seaborn as sns
color_mine_light = sns.set_hls_values(color = input_parsing.color_mine, h = None, l = 0.7, s = None)
sns.set(style="whitegrid", palette=[input_parsing.color_werk,color_mine_light],font_scale=0.2, rc={"lines.linewidth": 1})

ctr = 0
for machine in machine_names:
	functr = 0
	for func in funcs:
		print('Function: %s Machine : %s' % (func, machine))
		validation_data = input_parsing.read_validation_values("..",machine,func,1.2)	
		validation_pred = input_parsing.read_validation_predictions("..",machine,func,version)	
		validation_mixed = input_parsing.create_validation_set(validation_data, validation_pred, func)
		
		#Square set
		if func!= 'Daxpy':
			locs= [[1,1,1], [1,1,0], [1,0,0], [1,0,1], [0,1,1], [0,1,0], [0,0,1]]
			mid_sizes= [4096,8192,12288,16384]
			ctrs_fat = [2]#[2,4,5,6,7,8]
			ctrs_thin = [] #[5,7,10,13,16]
			square_validation_set = input_parsing.validation_set_split_BLAS3("Square",locs,mid_sizes,ctrs_fat,ctrs_thin,validation_mixed)
		else: 
			locs= [[1,1], [1,0], [0,1]]
			sizes= list(map(lambda y: y*1024*1024, [8,64,128,256]))
			square_validation_set = input_parsing.validation_set_split_BLAS1("Square",locs,sizes,validation_mixed)		
		input_parsing.create_statistics(square_validation_set, 'CoCopeLia_t', 'CoCopelia', 'werkhoven')

		#Fat&Thin set
		if func!= 'Daxpy':
			locs= [[1,1,1]]
			mid_sizes= [4096,8192,12288,16384]
			ctrs_fat = [3,4,5]
			ctrs_thin = [3,4,5]
			fathin_validation_set = input_parsing.validation_set_split_BLAS3("Square",locs,mid_sizes,ctrs_fat,ctrs_thin,validation_mixed)
		else: 
			locs= []
			sizes= []
			fathin_validation_set = input_parsing.validation_set_split_BLAS1("Square",locs,sizes,validation_mixed)		
		input_parsing.create_statistics(fathin_validation_set, 'CoCopeLia_t', 'CoCopelia', 'werkhoven')

		x_axis = 'T'

		merged_werk = pd.concat([square_validation_set, fathin_validation_set]).copy()
		merged_CoCo = pd.concat([square_validation_set, fathin_validation_set]).copy()

		merged_werk = merged_werk[merged_werk['T'] >= 4096]
		merged_werk['PE'] = merged_werk['PE_Comparisson']
		merged_werk['model'] = 'Werk. 2 cpe. [12]'

		merged_CoCo = merged_CoCo[merged_CoCo['T'] >= 4096]
		merged_CoCo['PE'] = merged_CoCo['PE_Mine']
		merged_CoCo['model'] = 'CoCopeLia eq. 7'

		merged_uni = pd.concat([merged_werk, merged_CoCo])
		merged_uni['func'] = func
		print(merged_uni.head(1))
		axs[functr,ctr].axhline(0, ls='--', linewidth = 0.6, color = 'gray', alpha=.7)
		axs[functr,ctr].axhline(20, ls='--', linewidth = 0.6, color = 'gray', alpha=.7)
		axs[functr,ctr].axhline(-20, ls='--', linewidth = 0.6, color = 'gray', alpha=.7)
		#axs[functr,ctr].grid(ls='--', linewidth = 0.6, color = 'gray')
		sns.violinplot(x=x_axis,y="PE",data=merged_uni, hue ='model', inner="box", linewidth = 1, ax=axs[functr,ctr])
		axs[functr,ctr].get_legend().remove()
		#axs[functr,ctr].set_xticks([])
		axs[functr,ctr].set_xlabel('')
		if (functr == 0):
			axs[functr,ctr].set_title(cool_names[ctr], fontsize=font)	

		if (ctr == 0):
			#axs[functr,ctr].set_ylabel(func)
			axs[functr,ctr].text( 2.6, 35.0, func, fontsize=font)
			if (functr == len(funcs)-1):
				axs[functr,ctr].text( 3, -125.0, 'Tile size (T)', fontsize=font)

		#axs[functr,ctr].set_ylabel(func)
		if ((functr == 1) & (ctr == 0)):
			axs[functr,ctr].text( -2, 0, 'Prediction Error (%)', fontsize=font, rotation = 90)
			#axs[functr,ctr].set_ylabel("Prediction Error (%)", fontsize=font)
		#elif (ctr == 0):
			#axs[functr,ctr].set_ylabel(func)
		#else:
		axs[functr,ctr].set_ylabel('')
		#if (functr == 2):
		#	axs[functr,ctr].set_xlabel(machine)
		#else:
		#	axs[functr,ctr].set_xlabel('')
		sns.despine()
		#sns_plot.savefig('Validation_plot_violins_sys-%s_func-%s_Xaxis-%s.pdf' % (func, machine,x_axis) )
		#



		functr += 1
	ctr += 1

#for ax in axs.flat:
#	ax.set(xlabel=machine_names, ylabel=funcs)

#Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
	ax.label_outer()
fig.set_size_inches(width, height)
# Create the legend
handles, labels = fig.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
	loc="lower center",   # Position of legend
	borderaxespad=0.1,    # Small spacing around legend box
	#title="Model",  # Title for the legend
	fontsize=font, fancybox = False, ncol=2
	)
fig.savefig('./Plots/Validation_plot_violins_all_Xaxis-%s.pdf' % (x_axis) )


