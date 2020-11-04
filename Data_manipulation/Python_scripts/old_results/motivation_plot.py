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

font=8
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=font) 
plt.rc('ytick', labelsize=font) 
plt.rc('axes', labelsize=font) 

# width as measured in inkscape
width = 3.487
height = width / 1.618

fig, ax = plt.subplots()
#fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
fig.subplots_adjust(left=.15, bottom=.18, right=.99, top=.97)

machine_num = 2

machines=["K40", "V100"] 
machine_names= ['dungani','silver1']

labels = ['cuBLAS Xt/Unified', '\(Ov_{pot}\)']
# Generate colors from the 'viridis' colormap
colors = cm.get_cmap('viridis',  4)
color_mine = colors(0)
color_serial = colors(1)
color_cublasxt = colors(2)
color_ideal = colors(3)

h2d_ti = []
h2d_tb = []
#h2d_sl = []
d2h_ti = []
d2h_tb = []
#d2h_sl = []

for machine in machine_names:
	infile_h2d = '../Results/%s/Models/transfer_model_0_-1.log' % machine
	infile_d2h = '../Results/%s/Models/transfer_model_-1_0.log' % machine

	with open(infile_h2d, 'r') as infile:
		h2d_log = infile.readlines()

	with open(infile_d2h, 'r') as infile:
		d2h_log = infile.readlines()

	h2d_ti.append (float(h2d_log[0]))
	h2d_tb.append (float(h2d_log[1]))
	#h2d_sl.append (float(h2d_log[2]))
	#print('ti = %e, tb = %e, sl = %lf' % (h2d_ti, h2d_tb, h2d_sl))

	d2h_ti.append (float(d2h_log[0]))
	d2h_tb.append (float(d2h_log[1]))
	#d2h_sl.append (float(d2h_log[2]))
	#print('ti = %e, tb = %e, sl = %lf' % (d2h_ti, d2h_tb, d2h_sl))

def transfer_h2d(bytes,idx):
	return h2d_ti[idx] + bytes* h2d_tb[idx]

def transfer_d2h(bytes,idx):
	return h2d_ti[idx] + bytes* h2d_tb[idx]

def GigaVal_per_s(val, time):
	return val * 1e-9 / time

def dgemm_flops(M,N,K):
	return M*N*(2*K+1)

def gemm_in_bytes(M,N,K,sz):
	return sz*(M*N + N*K + M*K)

def gemm_out_bytes(M,N,K,sz):
	return sz*(M*N)


cases = 5
case_names= ['DGEMM\n\(M,N,K=8K\)', 'DGEMM\n\(M,N,K=16K\)', 'DGEMM\n\(M,N=32K\)\n\(K=4K\)', 'SGEMM\n\(M,N,K=8K\)',  'DAXPY\n\(N=16^2K\)'] 				#'sgemm\nM,N,K=16K',

res_h2d = [[0.508762121,	transfer_h2d(gemm_in_bytes(16384,16384,16384,8),0),	transfer_h2d(gemm_in_bytes(32768,32768,4096,8),0),	0.260944694, 	transfer_h2d(2*16384*16384*8,0)]] 	#1.043672085, 	
res_d2h = [[0.160134524,	transfer_d2h(gemm_out_bytes(16384,16384,16384,8),0),	transfer_d2h(gemm_in_bytes(32768,32768,4096,8),0),	0.080121435, 	transfer_d2h(16384*16384*8,0)]] 	#0.320068687, 	
res_exec = [[0.881345,		7.038161993,						7.081064939,	0.347211152, 	3.280973e-02]]				# 2.730139256,		
achieved = [[1.278946877,	8.226300e+00,						9.432423443,	0.616209030, 	2.165699959]] 				# 3.389425039,		

res_h2d.append([transfer_h2d(gemm_in_bytes(8192,8192,8192,8),1),	0.520078719,	transfer_h2d(gemm_in_bytes(32768,32768,4096,8),1),		transfer_h2d(gemm_in_bytes(8192,8192,8192,4),1), 	transfer_h2d(2*16384*16384*8,1)]) 	#0.260036886,	
res_d2h.append([transfer_d2h(gemm_out_bytes(8192,8192,8192,8),1),	0.163283959,	transfer_d2h(gemm_in_bytes(32768,32768,4096,8),1),		transfer_d2h(gemm_out_bytes(8192,8192,8192,4),1), 	transfer_d2h(16384*16384*8,1)]) 	#0.081638977,	
res_exec.append([1.411715e-01,						1.123984933, 	1.128026962,							7.108644e-02, 						7.921952e-03]) 				#0.566234887,	
achieved.append([3.179269e-01, 						1.416376e+00, 	1.850381136,							1.657920e-01, 						0.545607090]) 				#7.097652e-01,	

print(res_h2d)
print(res_d2h)
ind = np.arange(cases)  # the x locations for the groups
bar_width = 0.15 # the width of the bars	
plt.grid('on', axis='y', linestyle='--')
slot_ind = 0.5/machine_num

plt.axhline(y=1,linewidth=1, color=color_serial)
#plt.text(0.5*ind[1] + 0.5*slot_ind - 0.05, 1.05, 'Serialized', rotation=90, fontsize=font, color=colors(0))
plt.text(ind[0] + slot_ind + bar_width -0.05, 0.92, 'Serial\nOffload', fontsize=font, color=color_serial)
for m in range(0,machine_num):
	ker_pot = []
	achieved_normalized = []
	for n in range(0,cases):
		h2d_t = res_h2d[m][n]
		d2h_t = res_d2h[m][n]
		exe_t = res_exec[m][n]
		zero_over = h2d_t + d2h_t + exe_t	
		full_over = max(h2d_t, d2h_t, exe_t)
		achieved_normalized.append(1 + (zero_over - achieved[m][n])/zero_over)
		ker_pot.append( 1 + (zero_over - full_over)/zero_over - achieved_normalized[n])
	if m == 0:
		plt.bar(ind + m*slot_ind, achieved_normalized, bar_width, label = labels[0], color=color_cublasxt)
		plt.bar(ind + m*slot_ind, ker_pot, bar_width, bottom = achieved_normalized, label = labels[1], color=color_ideal)
	else:
		plt.bar(ind + m*slot_ind, achieved_normalized, bar_width, color=color_cublasxt)
		plt.bar(ind + m*slot_ind, ker_pot, bar_width, bottom = achieved_normalized, color=color_ideal)

	plt.text(ind[0] + m*slot_ind-0.08, 1.6, '- ' + machines[m], rotation=90, fontsize=font)
	
	print(achieved)
	print(ker_pot)

#plt.ylabel('Normalized Performance')
#autolabel(rect)
#plt.xlabel('Problem')
plt.xticks(ind, case_names, fontsize=font-1 )# ,rotation=20)
plt.legend(fontsize=font, loc='upper center', fancybox = False, ncol=3) #,bbox_to_anchor=(0.5, 1.1))

#plt.title( "Flops_SpMV" )
#plt.savefig('motivation_plox.pdf', bbox_inches='tight')
#plt.close()


#x = np.arange(0.0, 3*np.pi , 0.1)
#plt.plot(x, np.sin(x))

ax.set_ylabel('Normalized Performance')
#ax.set_xlabel('Something (in unit)')
ax.set_ylim(0.5, 2)

fig.set_size_inches(width, height)
fig.savefig('motivation_plox.pdf')

