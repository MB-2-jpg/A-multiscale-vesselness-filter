import numpy as np
import Segmentation_filters as sf
import sklearn.metrics as slm

from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
"""
#this was meant to be a manual implementation but didn't seem to work properly
def recall(treshhold, Y):
    p=len([Y>treshhold])
    n = len(Y)
    return p/n

def prior(X,Y):
    return (len(X)/len(Y))
def precision(treshhold ,X,Y):
    false_pos_proportion_in_the_negatve = len([X>treshhold])/len(X)
    print(f"false pos = {false_pos_proportion_in_the_negatve}")
    prior_probability_estimation =prior(X,Y)
    recall_treshhold =recall(treshhold,Y)
    return prior_probability_estimation*recall_treshhold/((prior_probability_estimation*recall_treshhold)+((1-prior_probability_estimation)*false_pos_proportion_in_the_negatve))

def AUPRC(X,Y,number_of_treshholds=50):
    s=0
    recall_n=1
    for number in range(number_of_treshholds):
       
        treshhold=(number+1)/number_of_treshholds
        recall_n1=recall(treshhold,Y)
        precision_n=precision(treshhold ,X,Y)
        s+=precision_n*(recall_n-recall_n1)
        recall_n=recall_n1
        print(f"recall {recall_n}")
        print(f"precision {precision_n}")

    return s
"""
list_of_results=[]
for i in range(10):
    segmentated_image= np.array(sf.main_segmentation(chosen_tau=0.1*i))



    img = Image.open("39_manual1.gif")

            # Dimensions
            #print(img.size, img.mode)

            # Lire pixels
    pixels = img.load()


    pixels_original = img.copy().load()
    #ACP= img.load()
    x, y = img.size
    si=np.zeros(x*y)

    #Niveau de gris
    #gris = np.zeros((x,y))
    print(img.size, img.mode)
    gris= np.zeros(x*y)
    result=np.zeros((x,y))
    array_of_pixel_values=np.zeros((y,x,3))

        
    for i in range(x):
        for j in range(y):

            
            gris[i*j +i] = pixels[i, j]
            si[i*j+i]=segmentated_image[i,j]
            
    print(gris)
    print(np.max(gris))
    print(np.max(segmentated_image))
    segmentated_image=(255/np.max(segmentated_image))*segmentated_image
    print(segmentated_image)
    AUPRC=slm.average_precision_score(gris, si, pos_label=255)
    list_of_results.append(AUPRC)
    print(f"AUPRC {AUPRC}")

list_of_results=np.log(np.array(list_of_results))
tau_values=np.zeros(10)
for i in range(10):
      tau_values[i]=i*0.1
plt.plot(tau_values,list_of_results)
plt.xlabel("$\tau$ values")
plt.ylabel("AUPRC")
plt.show()