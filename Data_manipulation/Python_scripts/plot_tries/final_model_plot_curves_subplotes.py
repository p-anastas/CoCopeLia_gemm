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
color_werk = colors1(2)

machine_names= ['testbed-II_Tesla-V100']
funcs=['Dgemm', 'Sgemm']
version='1.2'
dim_list=[[4096,4096,4096],[8192,8192,8192],[16384,16384,16384]]
loc=[1,1,1]
mindim=512

fig, ax = plt.subplots(len(funcs)*len(machine_names)*len(dim_list), sharex=True)
#fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
fig.subplots_adjust(left=.15, bottom=.17, right=.99, top=.97)

ctr = 0 
for func in funcs:
	#for machine in machine_names:
	for dim in dim_list:
		maxdim=min(min(dim),max(dim)/1.5) # min(dim)/2
		machine = machine_names[0]
		validata_CoCo = pd.read_csv('../Results/%s/validation/CoCopeLia_%s_0_v%s.log' % (machine, func, version),
							   header =None, usecols = [0,1,2,3,4,5,6,8,9,10,11], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc',  'CoCopeLia_av_t' ,  'CoCopeLia_min_t' ,  'CoCopeLia_max_t', 'CoCopeLia_first_t'])
		validata_cuBLASXt = pd.read_csv('../Results/%s/validation/cuBLASXt_%s_0.log' % (machine, func),
							   header =None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'cublasXt_t'])

		validata = pd.merge(validata_CoCo, validata_cuBLASXt, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])
		
		#print(validata.head(1))
		#print(validata['CoCopeLia_t'].min())

		validata_plot = validata[(validata['M'] == dim[0]) & (validata['N'] == dim[1]) & (validata['K'] == dim[2])
								 & (validata['Aloc'] == loc[0]) & (validata['Bloc'] == loc[1]) &(validata['Cloc'] == loc[2])]

		streamed_list = list(validata_plot[(validata_plot['T'] >= mindim) & (validata_plot['T'] <= maxdim)]['CoCopeLia_av_t'])
		streamed_list_flops_av = list(map(lambda x: GigaVal_per_s(dgemm_flops(dim[0],dim[1],dim[2]),x*1024),streamed_list))

		cublasXt_list = list(validata_plot[(validata_plot['T'] >= mindim) & (validata_plot['T'] <= maxdim)]['cublasXt_t'])
		cublasXt_list_flops = list(map(lambda x: GigaVal_per_s(dgemm_flops(dim[0],dim[1],dim[2]),x*1024),cublasXt_list))

		val_tile_points = list(validata_plot[(validata_plot['T'] >= mindim) & (validata_plot['T'] <= maxdim)]['T'])


		pred_data_avg = pd.read_csv('../Results/%s/validation/%s_CoCopelia_predict_avg_0_v%s.log' % (machine, func, version),
								header =None,  names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia'])

		pred_data_min = pd.read_csv('../Results/%s/validation/%s_CoCopelia_predict_min_0_v%s.log' % (machine, func, version),
								header =None,  names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia'])

		pred_data_max = pd.read_csv('../Results/%s/validation/%s_CoCopelia_predict_max_0_v%s.log' % (machine, func, version),
								header =None,  names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia'])

		#pred_data = pred_data_avg

		#print(pred_data.head(1))
		#print(pred_data['CoCopelia'].max())

		pred_plot_avg = pred_data_avg[(pred_data_avg['M'] == dim[0]) & (pred_data_avg['N'] == dim[1]) & (pred_data_avg['K'] == dim[2])
								 & (pred_data_avg['Aloc'] == loc[0]) & (pred_data_avg['Bloc'] == loc[1]) &(pred_data_avg['Cloc'] == loc[2])]

		werk_list = list(pred_plot_avg[(pred_plot_avg['T'] >= mindim) & (pred_plot_avg['T'] <= maxdim) & (pred_plot_avg['T'] <= max(val_tile_points))]['werkhoven'])
		werk_list_flops = list(map(lambda x: GigaVal_per_s(dgemm_flops(dim[0],dim[1],dim[2]),x*1024),werk_list))
		CoCo_list = list(pred_plot_avg[(pred_plot_avg['T'] >= mindim) & (pred_plot_avg['T'] <= maxdim) & (pred_plot_avg['T'] <= max(val_tile_points))]['CoCopelia'])
		CoCo_list_flops = list(map(lambda x: GigaVal_per_s(dgemm_flops(dim[0],dim[1],dim[2]),x*1024),CoCo_list))

		pred_tile_points = list(pred_plot_avg[(pred_plot_avg['T'] >= mindim) & (pred_plot_avg['T'] <= maxdim) & (pred_plot_avg['T'] <= max(val_tile_points))]['T'])

		#print(validata_plot)
		print(val_tile_points)
		print(streamed_list)
		#print(cublasXt_list)

		#print(pred_plot)
		print(pred_tile_points)
		print(werk_list)
		print(CoCo_list)
		ax[ctr].plot(pred_tile_points, werk_list,
				 linewidth=1, color=color_werk)#, label='Werk.')
		ax[ctr].plot(val_tile_points, streamed_list,
				 '.', markersize=2, color=color_mine)#, label='y_val')
		ax[ctr].plot(pred_tile_points, CoCo_list,
				 linewidth=1, color=color_mine)#, label='Eq. 8')
		#plt.plot(val_tile_points, cublasXt_list_flops,
		#		 '.', markersize=2, color=color_cublasxt, label='cuBLASXt')

		pred_plot_min = pred_data_min[(pred_data_min['M'] == dim[0]) & (pred_data_min['N'] == dim[1]) & (pred_data_min['K'] == dim[2])
								 & (pred_data_min['Aloc'] == loc[0]) & (pred_data_min['Bloc'] == loc[1]) &(pred_data_min['Cloc'] == loc[2])]

		pred_plot_max = pred_data_max[(pred_data_max['M'] == dim[0]) & (pred_data_max['N'] == dim[1]) & (pred_data_max['K'] == dim[2])
								 & (pred_data_max['Aloc'] == loc[0]) & (pred_data_max['Bloc'] == loc[1]) &(pred_data_max['Cloc'] == loc[2])]



		streamed_list_min = list(validata_plot[(validata_plot['T'] >= mindim) & (validata_plot['T'] <= maxdim)]['CoCopeLia_min_t'])
		streamed_list_flops_min = list(map(lambda x: GigaVal_per_s(dgemm_flops(dim[0],dim[1],dim[2]),x*1024),streamed_list_min))

		streamed_list_max = list(validata_plot[(validata_plot['T'] >= mindim) & (validata_plot['T'] <= maxdim)]['CoCopeLia_max_t'])
		streamed_list_flops_max = list(map(lambda x: GigaVal_per_s(dgemm_flops(dim[0],dim[1],dim[2]),x*1024),streamed_list_max))
		
		ax[ctr].fill_between(val_tile_points, streamed_list_min, streamed_list_max, alpha=0.2)

		#CoCo_list_min = list(pred_plot_max[(pred_plot_max['T'] >= mindim) & (pred_plot_max['T'] <= maxdim)]['CoCopelia'])
		#CoCo_list_flops_min = list(map(lambda x: GigaVal_per_s(dgemm_flops(dim[0],dim[1],dim[2]),x*1024),CoCo_list_min))
		#CoCo_list_max = list(pred_plot_min[(pred_plot_min['T'] >= mindim) & (pred_plot_min['T'] <= maxdim)]['CoCopelia'])
		#CoCo_list_flops_max = list(map(lambda x: GigaVal_per_s(dgemm_flops(dim[0],dim[1],dim[2]),x*1024),CoCo_list_max))
		#ax.fill_between(pred_tile_points, CoCo_list_flops_min, CoCo_list_flops_max, alpha=0.2)

		#t_serial = 1.828934
		#plt.axhline(y=GigaVal_per_s(dgemm_flops(dim[0],dim[1],dim[2]),1024*1.828934), linestyle='--', linewidth=0.5, color=colors(1))
		#plt.text(1024, GigaVal_per_s(dgemm_flops(dim[0],dim[1],dim[2]),1024*1.828934)*0.75,'Serial\nOffload', fontsize=font, color=colors(1))
		#autolabel(rect)
		#autolabel(rect)
		#plt.legend(fontsize=font, loc='upper center', fancybox = False, ncol=3)
	
		ctr += 1

#ax.set_ylabel('Time (s)')
		#ax.set_yscale('log')
#ax.set_xlabel('Tile Dim (T)')

		#ax.set_ylim(1,10)

for ax in fig.get_axes():
    ax.label_outer()

fig.set_size_inches(width, height)
fig.savefig('final_model_curves_dim-%d-%d-%d_loc-%d-%d-%d.pdf' % (dim[0], dim[1], dim[2], loc[0], loc[1], loc[2]))

