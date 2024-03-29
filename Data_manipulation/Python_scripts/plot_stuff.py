#from pylab import cm
# Generate colors from the 'viridis' colormap
#colors = cm.get_cmap('viridis',  4)
#colors1 = cm.get_cmap('magma',  4)
#color_mine = colors(0)
#color_cublasxt = colors(2)
#color_ideal = colors(3)
#color_werk = colors1(2)
#colors2 = cm.get_cmap('magma',  6)

import matplotlib
import seaborn as sns
#Red
red1= sns.color_palette("Reds_d",1)
red2= sns.color_palette("Reds_d",2)
red3= sns.color_palette("Reds_d",3)
red4= sns.color_palette("Reds_d",4)
red5= sns.color_palette("Reds_d",5)
red6= sns.color_palette("Reds_d",6)
red7= sns.color_palette("Reds_d",7)
red8= sns.color_palette("Reds_d",8)

#YellowGreen
yg1= sns.color_palette("YlGn_d",1)
yg2= sns.color_palette("YlGn_d",2)
yg3= sns.color_palette("YlGn_d",3)
yg4= sns.color_palette("YlGn_d",4)
yg5= sns.color_palette("YlGn_d",5)
yg7= sns.color_palette("YlGn_d",7)

#GreenBlue
gb1= sns.color_palette("GnBu_d",1)
gb2= sns.color_palette("GnBu_d",2)
gb3= sns.color_palette("GnBu_d",3)
gb4= sns.color_palette("GnBu_d",4)
gb5= sns.color_palette("GnBu_d",5)
gb6= sns.color_palette("GnBu_d",6)
gb7= sns.color_palette("GnBu_d",7)
gb8= sns.color_palette("GnBu_d",8)

cp2 = list(map(lambda x: sns.desaturate(x,0.9),[red7[2],gb7[4]]))
cp2v1 = list(map(lambda x: sns.desaturate(x,0.9),[red7[2],yg7[0]]))
cp3 = list(map(lambda x: sns.desaturate(x,0.9),[yg7[0],gb7[4],red7[2]]))
cp4 = list(map(lambda x: sns.desaturate(x,0.9),red1+gb2+yg1))
cp2_2_inter = list(map(lambda x: sns.desaturate(x,0.9),red7 + gb7))
cp2_2 = list(map(lambda x: sns.desaturate(x,0.9),[red7[0],red7[3],gb7[4],gb7[6]]))
cp_total_spectrum = list(map(lambda x: sns.desaturate(x,0.9),gb7 + yg7 + red7))
#sns.set_palette(cp4)
print (['%02x%02x%02x' % (int(e[0]*256),int(e[1]*256),int(e[2]*256)) for e in cp_total_spectrum])
colors2 = matplotlib.colors.ListedColormap(cp2, name='2_colours')
colors3 = matplotlib.colors.ListedColormap(cp3, name='3_colours')
colors4 = matplotlib.colors.ListedColormap(cp4, name='4_colours')
colors2_2 = matplotlib.colors.ListedColormap(cp2_2, name='2_2_colours')
#colors2 = matplotlib.colors.ListedColormap(sns.color_palette().as_hex())
