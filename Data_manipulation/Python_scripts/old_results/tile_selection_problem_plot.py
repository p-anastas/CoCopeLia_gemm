def autolabel(rects):
	"""
	Attach a text label above each bar displaying its height
	"""
	for rect in rects:
		height = float(rect.get_height())
		ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
		'%.4f' % float(height),
		ha='center', va='bottom')


import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from pylab import cm
import csv

def GigaVal_per_s(val, time):
	return val * 1e-9 / time

def dgemm_flops(M,N,K):
	return M*N*(2*K+1)

font=8
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=font) 
plt.rc('ytick', labelsize=font) 
plt.rc('axes', labelsize=font) 

# width as measured in inkscape
width = 3.487
height = width / 1.618 *3/4

fig, ax = plt.subplots()
#fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
fig.subplots_adjust(left=.15, bottom=.21, right=.99, top=.99)

machine_num = 2

machines=["K40", "V100"] #
machine_names= ['dungani','silver1'] #

dgemm_list = []
sgemm_list = []
for machine in machine_names:
	valfile_dgemm = '../Results/%s/validation/CoCoBLAS_dgemm_tile_0.log' % machine
	valfile_sgemm = '../Results/%s/validation/CoCoBLAS_sgemm_tile_0.log' % machine

	with open(valfile_dgemm, 'r') as infile:
		reader = csv.reader(infile)
		dgemm_log = list(reader)

	with open(valfile_sgemm, 'r') as infile:
		reader = csv.reader(infile)
		sgemm_log = list(reader)

	dgemm_list.append (dgemm_log)
	sgemm_list.append (sgemm_log)



cases = 3		
case_sizes= [[16384,16384,16384],[8192,8192,8192], [4096,4096,4096]]
case_machines = ["V100", "V100", "V100"] #["K40","K40","K40"] #
case_locs = [[1,1,1], [1,1,0], [0,1,1]] # [[1,1,1], [1,1,0], [1,1,1]] #
# Generate colors from the 'viridis' colormap
#colors = cm.get_cmap('viridis', cases + 1)
colors = cm.get_cmap('magma', cases + 1 )

floplist=[[1.4e3,4.3e3], [7e3,14e3]]

#print(dgemm_list[0][0])
#print(sgemm_list[0][0])

#plt.grid('on', axis='y', linestyle='--')
#plt.axhline(y=1,linewidth=1, color=colors(0))
#plt.text(0.5*ind[1] + 0.5*slot_ind - 0.05, 1.05, 'Serialized', rotation=90, fontsize=font, color=colors(0))
for n in range(0,cases):
	case_list_val = []
	case_list_tile = []	
	maxT = 0 
	maxVal = 0 
	if case_machines[n] == "K40":
		dgemm_curlog = dgemm_list[0]
		m = 0
	elif case_machines[n] == "V100":
		dgemm_curlog = dgemm_list[1]
		m = 1
	for elem in dgemm_curlog:
		if (int(elem[0]) == case_sizes[n][0] and int(elem[1]) == case_sizes[n][1] and int(elem[2]) == case_sizes[n][2] and int(elem[3]) == int(elem[4]) and int(elem[3]) > 512 
		and int(elem[3]) <= 8192 and int(elem[5]) == case_locs[n][0] and int(elem[6]) == case_locs[n][1] and int(elem[7]) ==case_locs[n][2]):
			case_list_tile.append(int(elem[3]))
			case_list_val.append(GigaVal_per_s(dgemm_flops(case_sizes[n][0],case_sizes[n][1],case_sizes[n][2]), float(elem[-3]))/1024)
			if (GigaVal_per_s(dgemm_flops(case_sizes[n][0],case_sizes[n][1],case_sizes[n][2]), float(elem[-3]))/1024 > maxVal):
				maxVal = GigaVal_per_s(dgemm_flops(case_sizes[n][0],case_sizes[n][1],case_sizes[n][2]), float(elem[-3]))/1024
				maxT = int(elem[3])
				#print('Tile : %d -> t = %lf' %(int(elem[3]), float(elem[-3])))
	if n == 0:
		type = 'o'
	elif n == 1:
		type = '^'
	else:
		type = 's'
	plt.plot(case_list_tile, case_list_val, type, markersize =4, color = colors(n), label = '\(Dim=' + str(int(case_sizes[n][0]/1024)) + 'K\)') #case_machines[n] + '-dgemm-M,N,K=' + str(int(case_sizes[n][0]/1024)) + 'K')
	plt.axvline(x=maxT,linewidth=0.5, linestyle = '--', color=colors(n))
	plt.text(maxT+ 120, 2.2, str(maxT), rotation=90, fontsize=font, color=colors(n)) # 0.42
 

#autolabel(rect)
plt.legend(fontsize=font, loc='lower right', fancybox = False)

ax.set_ylabel('Performance ( Tflops/s )')
ax.set_xlabel('Tile size (T)')
#ax.set_ylim(0.4, 1.19)
ax.set_ylim(2, 7.2)

fig.set_size_inches(width, height)
fig.savefig('problem_plox_silver1_smol.pdf')
#fig.savefig('problem_plox_dun_smol.pdf')

	#	case_list_val = []
	#	case_list_tile = []
	#	for elem in sgemm_curlog:
	#		if (int(elem[0]) == case_sizes[n][0] and int(elem[1]) == case_sizes[n][1] and int(elem[2]) == case_sizes[n][2] and int(elem[3]) == int(elem[4]) and int(elem[3]) >= 512 and int(elem[3]) <= 8192 and int(elem[5]) == int(elem[6]) and int(elem[6]) == int(elem[7]) and int(elem[7]) == 1):
	#			case_list_tile.append(int(elem[3]))
	#			case_list_val.append(GigaVal_per_s(dgemm_flops(case_sizes[n][0],case_sizes[n][1],case_sizes[n][2]), float(elem[-3]))/floplist[1])
				#print('Tile : %d -> t = %lf' %(int(elem[3]), float(elem[-3])))
	#	plt.plot(case_list_tile, case_list_val, color = colors(cases+n), label = machines[m] +'s' + case_names[n])


