import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib as mpl

from skimage import exposure 
from skimage.measure import label, regionprops

def plot_imshow(data, save_file):
	font_size = 14 # print
	fig = plt.figure()  # a new figure window
	ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
	data = data[:, :, 0:3]
	ax.imshow(data)
	ax.xaxis.set_label_position('bottom') 
	ax.xaxis.tick_bottom()
	ax.set_xlabel('x (m)', fontdict={'fontsize': font_size})
	ax.set_ylabel('y (m)', fontdict={'fontsize': font_size})
	plt.title('Raw PNG with RGB')
	plt.tight_layout()
	plt.savefig('image_out/'+save_file, format='svg', transparent=True)

def clip(model, perc):
	(ROWs, COLs) = model.shape
	reshape2D_1D = model.reshape(ROWs*COLs)
	reshape2D_1D = np.sort(reshape2D_1D)
	if perc != 100:
		min_num = reshape2D_1D[ round(ROWs*COLs*(1-perc/100)) ]
		max_num = reshape2D_1D[ round((ROWs*COLs*perc)/100) ]
	elif perc == 100:
		min_num = min(model.flatten())
		max_num = max(model.flatten())
	return max_num, min_num, 

def plot_rgb(data_tiff, clip, save_file):
	blue = data_tiff[:, :, 2-1]
	green = data_tiff[:, :, 3-1]
	red = data_tiff[:, :, 4-1]
	stackedRGB = np.stack((red, green, blue), axis=2)
	# clip color 
	pLow, pHigh = np.percentile(stackedRGB[~np.isnan(stackedRGB)], (clip, 100-clip))
	img_rescale = exposure.rescale_intensity(stackedRGB, in_range=(pLow, pHigh))
	plt.imshow(img_rescale, clim=(0, 1))
	plt.tight_layout()
	plt.savefig('image_out/'+save_file, format='svg', transparent=True)
	plt.show()

def cleaning_data(data_tiff): #? cleaning data: remove nan and inf
	data_tiff = np.nan_to_num(data_tiff)
	# print('max value = ', data_tiff.max(), 'min value = ', data_tiff.min())
	return data_tiff, data_tiff.min(), data_tiff.max()

def plot_cluster(data, num_band, save_file):
	fig = plt.figure(1)  # a new figure window
	font_size = 14 # print
	ax1 = plt.subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
	ax1.xaxis.set_label_position('bottom') 
	ax1.xaxis.tick_bottom()
	ax1.set_xlabel('x (m)', fontdict={'fontsize': font_size})
	ax1.set_ylabel('y (m)', fontdict={'fontsize': font_size})
	ax1.set_title('before cleaning of band ' + str(num_band))
	# colorbar start here
	#ffff52	(yellow) Bare Soils
	#10d22c	(green) Vegetation
	#0000ff	(blue) Water
	# cmap = mpl.colors.ListedColormap(['black', 'blue', 'gray'])
	# cax = ax.imshow(data, cmap=cmap, vmin=min_num, vmax=max_num)
	cmap = mpl.colors.ListedColormap(['#0000ff', '#10d22c'])
	cax = ax1.imshow(data, cmap=cmap)

	cbar = fig.colorbar(cax, orientation='vertical', fraction=0.025, pad=0.01, shrink=0.9)
	cbar.set_label('clusters', labelpad=18, rotation=270, 
					fontdict={'fontsize': font_size, 'fontweight': 'bold'})
	for l in cbar.ax.yaxis.get_ticklabels():
		l.set_fontsize(10)
		l.set_weight('bold')
		# cbar.set_ticks([0.057, 0.0575, 0.058, 0.0585, 0.059])
		# cbar.set_ticklabels([2.0, 3.0, 4.0])
	# colorbar end here
	plt.tight_layout()
	plt.savefig('image_out/'+save_file, format='svg', transparent=True)
	plt.show()

def convert_array_to_binary(kmeans):
	temp = np.unique(kmeans)
	if len(temp) == 2:
		min_num, _ = np.unique(kmeans)
		binary = np.where(kmeans > min_num, 0, 1)
	else:
		print('please fix me')
	return binary

def image_segmentation(kmeans, min_pixel, max_pixel, save_file):
	# 0 is soil and 1 is water
	# binary = np.where(kmeans >= kmeans.max()/2, 0, 1)
	binary = convert_array_to_binary(kmeans)
	#? begin image segmentation
	label_out = label(binary, connectivity=1, return_num=False)
	for region in regionprops(label_out):
		(min_row, min_col, max_row, max_col) = region.bbox
		if region.area >= min_pixel and region.area <= max_pixel:
			binary[min_row:max_row, min_col:max_col] = 0
	################################# for QC plot
	# plt.figure(1)  
	# plt.imshow(binary)
	# plt.title('after cleaning')
	# plt.show()
	##################################
	plt.tight_layout()
	plt.savefig('image_out/'+save_file, format='svg', transparent=True)
	return binary

def count_number_of_pixel(cleaned_kmeans):
	# 0 is soil and 1 is water
	binary = np.where(cleaned_kmeans >= cleaned_kmeans.max()/2, 0, 1).flatten()
	count = 0
	for i in range (0, len(binary)):
		if binary[i] != 0.:
			count += 1
	return count
