import Vesselness_filter as Vf
import numpy as np

def main_3D_segmentation(image3D):
    shape=image3D.shape
    number_of_images=shape[2]
    segmentation_of_3D_image=np.zeros(shape)
    for i in range(number_of_images):
          segmentation_of_3D_image[:,:,i]=Vf.main_segmentation(input_as_array=True , pixels_array=image3D[:,:,i])


    return segmentation_of_3D_image