import pandas as pd
import math
import statistics

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from pylab import cm

def GigaVal_per_s(val, time):
	return val * 1e-9 / time

def dgemm_flops(M,N,K):
	return M*N*(2*K+1)

def transfer_h2d(bytes,idx):
	return h2d_ti[idx] + bytes* h2d_tb[idx]

def transfer_d2h(bytes,idx):
	return h2d_ti[idx] + bytes* h2d_tb[idx]


font = 8
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
fig.subplots_adjust(left=.15, bottom=.17, right=.99, top=.97)


# Generate colors from the 'viridis' colormap
colors = cm.get_cmap('viridis',  4)
colors1 = cm.get_cmap('magma',  4)
color_mine = colors(0)
color_cublasxt = colors(2)
color_ideal = colors(3)
color_werk = colors1(2)

machine_names= ['silver1']

for machine in machine_names:
	infile_h2d = '../Results/%s/Models/transfer_model_0_-1.log' % machine
	infile_d2h = '../Results/%s/Models/transfer_model_-1_0.log' % machine

	with open(infile_h2d, 'r') as infile:
		h2d_log = infile.readlines()

	with open(infile_d2h, 'r') as infile:
		d2h_log = infile.readlines()

	h2d_ti = (float(h2d_log[0]))
	h2d_tb = (float(h2d_log[1]))
	h2d_sl = (float(h2d_log[2]))
	print('ti = %e, tb = %e, sl = %lf' % (h2d_ti, h2d_tb, h2d_sl))

	d2h_ti = (float(d2h_log[0]))
	d2h_tb = (float(d2h_log[1]))
	d2h_sl = (float(d2h_log[2]))
	print('ti = %e, tb = %e, sl = %lf' % (d2h_ti, d2h_tb, d2h_sl))

def transfer_h2d(bytes):
	return h2d_ti + bytes* h2d_tb

def transfer_d2h(bytes):
	return h2d_ti + bytes* h2d_tb

def transfer_h2d_bid(bytes):
	return h2d_ti + bytes* h2d_tb*h2d_sl

def transfer_d2h_bid(bytes):
	return h2d_ti + bytes* h2d_tb*d2h_sl

def transfer_over(bytes1,bytes2,direction):
	if direction == 0:
		actual_over_t = min(transfer_h2d_bid(bytes1), transfer_d2h_bid(bytes2))
		if actual_over_t == transfer_h2d_bid(bytes1):
			lo_ti, lo_tb, lo_sl, bytes_lo = d2h_ti, d2h_tb,d2h_sl, bytes2
		else:
			lo_ti, lo_tb, lo_sl, bytes_lo = h2d_ti, h2d_tb,h2d_sl, bytes1
		#print(actual_over_t)
		total_over = actual_over_t *(1 - 1/lo_sl) + bytes_lo* lo_tb  + lo_ti /lo_sl
		#print(total_over)
		return total_over

for machine in machine_names:

	h2d_validata = pd.read_csv('../Results/%s/validation/bid_transfer_benchmark_0_-1_10000-10000000-1000.log' % (machine),
							   header =None, usecols = [0,1,2,3], names= ['bytes', 'b2_sqb1_t', 'b2_eq_b1_t','b2_eq_b1div2_t' ])
	#print(h2d_validata.head(1))
		#print(validata['streamed_t'].max())
	d2h_validata = pd.read_csv('../Results/%s/validation/bid_transfer_benchmark_-1_0_10000-10000000-1000.log' % (machine),
						   header=None, usecols=[0, 1, 2, 3],
						   names=['bytes', 'b2_sqb1_t', 'b2_eq_b1_t', 'b2_eq_b1div2_t'])
	#print(d2h_validata.head(1))
		#print(pred_data['CoCopelia'].max())
	for scenario in [(h2d_validata['bytes'] < 1024*1024) & (h2d_validata['bytes'] > 1024*100), h2d_validata['bytes'] >= 1024*1024]:
		validata_plot = h2d_validata[scenario]  # [(validata['M'] == 16384) & (validata['N'] == 16384) & (validata['K'] == 16384)
	# & (validata['Aloc'] == 1) & (validata['Bloc'] == 1) &(validata['Cloc'] == 1)]
		print('Samples: %d' % len(validata_plot))

		b2_sqb1_t_list = list(validata_plot['b2_sqb1_t'])
		b2_eq_b1_t_list = list(validata_plot['b2_eq_b1_t'])
		b2_eq_b1div2_t_list = list(validata_plot['b2_eq_b1div2_t'])
		byte_points = list(validata_plot['bytes'])

		#plt.plot(byte_points,b2_sqb1_t_list,\
		#		'.', markersize=1, color=color_ideal, label='\(b2=sqrt{b1}\)')
		#plt.plot(byte_points,b2_eq_b1_t_list,\
		#		's', markersize=1, color=color_ideal, label='\(b2 = b1\)')
		#plt.plot(byte_points,b2_eq_b1div2_t_list,\
		#		'^', markersize=1, color=color_ideal, label='\(b2 = b1/2\)')
		for valist in [b2_sqb1_t_list,b2_eq_b1_t_list,b2_eq_b1div2_t_list]:
			prev_transfers = list(map(lambda x: transfer_h2d(x), byte_points))
			#print(prev_transfers)
			plt.plot(byte_points, list(map(lambda x,y : abs((y-x)/y), prev_transfers,valist)) ,
					linewidth=1, color=color_werk, label='Previous')
			if valist == b2_sqb1_t_list:
				lam = lambda x: math.sqrt(x)
			elif valist == b2_eq_b1_t_list:
				lam = lambda x: x
			else:
				lam = lambda x: x/2
			my_transfers = list(map(lambda x,y: transfer_over(x,y,0), byte_points, list(map(lam,byte_points))))
			plt.plot(byte_points, list(map(lambda x,y : abs((y-x)/y), my_transfers,valist)) ,
					linewidth=1, color=color_mine, label='Mine')
			print('Machine : %s' % (machine))
			MAPE_prev = statistics.mean(list(map(lambda x,y : 100*abs((y-x)/y), prev_transfers,valist)))
			MAPE_min = statistics.mean(list(map(lambda x,y : 100*abs((y-x)/y), my_transfers,valist)))
			print( "Previous MAPE : %lf" % MAPE_prev)
			print( "Bidirectional MAPE : %lf" % MAPE_min)

		plt.legend(fontsize=font-2, loc='upper center', fancybox=False, ncol=3)

		ax.set_ylabel('Error')
		ax.set_xlabel('Transfer Size (Bytes)')
		ax.set_yscale('log')
		#ax.set_ylim(1, 10)

		fig.set_size_inches(width, height)
		#fig.savefig('bidirectional_comp.pdf')







