import core as c
import skimage.io as skio

from sklearn.cluster import KMeans

# TODO import data
# Sentinel-2 MSI: MultiSpectral Instrument, Level-2A
stacked_images = skio.imread('input_images/Chi_10m_12bands.tif', plugin="tifffile")
# print('check image size', stacked_images.shape)

# TODO cleaning data contained NaN
stacked_images, _, _ = c.cleaning_data(stacked_images)

# TODO plot images
# c.plot_imshow(stacked_images, 'test.svg')
c.plot_rgb(stacked_images, 5, 'test.svg')
sss
# TODO save rgb image
# folder = 'image_out/'
# name = 'image'
# # filepath = 'image_out/' + 'image' + (str(1).zfill(3)) + '.jpg'
# # print('save rgb as jpg:', filepath)
# # c.save_rgb_jpg(stacked_images, 2, filepath, fig=None)

# '''
# keep in mind the true band order might not follow this sequence
# 0  B1	Aerosols			60 meters (bad)
# 1  B2	Blue			    10 meters (moderate)
# 2  B3	Green			    10 meters (moderate)
# 3  B4	Red			        10 meters (moderate)
# 4  B5	Red Edge 1			20 meters (moderate)
# 5  B6	Red Edge 2			20 meters (good)
# 6  B7	Red Edge 3			20 meters (good)
# 7  B8	NIR			        10 meters (good)
# 8  B8A	Red Edge 4			20 meters (good)
# 9  B9	Water vapor			60 meters (good)
# 10 B11	SWIR 1			    20 meters (moderate)
# 11 B12	SWIR 2			    20 meters (moderate)	
# '''

# # TODO loop save each band
# num_band = (5, 6, 7, 8, 9) # using only good bands

# # TODO K-means
# for i in num_band:

# 	images_2D = stacked_images[:, :, i]
# 	vector_data = images_2D.reshape(-1, 1) # flatten matrix to vector for each band (row * col, 3) 
# 	number_of_classes = 2
# 	random_centroid = 42 # interger number range 0-42
# 	kmeans = KMeans(n_clusters = number_of_classes, random_state = random_centroid).fit(vector_data)
# 	kmeans = kmeans.cluster_centers_[kmeans.labels_]
# 	kmeans = kmeans.reshape(images_2D.shape)
# 	c.plot_cluster(kmeans, i, 'test.svg')

# 	# TODO remove unwanted areas using image segmentation
# 	if i == 9:
# 		min_pixel, max_pixel = 1, 150
# 		cleaned_kmeans = c.image_segmentation(kmeans, min_pixel, max_pixel, 'test.svg')
# 	else:
# 		min_pixel, max_pixel = 1, 100
# 		cleaned_kmeans = c.image_segmentation(kmeans, min_pixel, max_pixel, 'test.svg')

# 	# TODO count pixel
# 	print('water pixel of band', i, ':',  c.count_number_of_pixel(cleaned_kmeans))
