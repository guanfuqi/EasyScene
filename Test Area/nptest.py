import numpy as np
pano_h = 512
pano_w = 1024
x, y = np.meshgrid(np.arange(pano_w),np.arange(pano_h), indexing = 'xy')
pixel_coord = np.stack((x,y),axis = -1)
print(pixel_coord.shape)