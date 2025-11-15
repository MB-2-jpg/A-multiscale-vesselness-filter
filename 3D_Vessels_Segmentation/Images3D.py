# Source - https://stackoverflow.com/a
# Posted by Dirco, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-08, License - CC BY-SA 4.0
# import sample data
from skimage.data import cells3d
import Segmentation3D as S3D
import napari

# create a `Viewer` and `Image` layer here

import numpy as np
import nrrd
from PIL import Image
import numpy as np
filename = "001.nrrd"
readdata, header = nrrd.read(filename)
print(readdata.shape) # (512, 512, 504)

img = np.array(readdata[:,:,:])
seg_img=S3D.main_3D_segmentation(img)
viewer, image_layer = napari.imshow(seg_img)
# print shape of image data
print(image_layer.data.shape)
#viewer.add_image(nuclei)
# start the event loop and show the viewer
napari.run()