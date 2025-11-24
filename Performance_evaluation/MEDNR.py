"""import numpy as np
import Segmentation_filter as sf


def MEDNR(segmented_image):
    M=np.max(segmented_image)
    segmented_image = segmented_image/M
    tresh=0.6
    response_segmented_region=segmented_image[segmented_image>tresh]
    return np.median(response_segmented_region)
#print(MEDNR(segmented_image))
"""