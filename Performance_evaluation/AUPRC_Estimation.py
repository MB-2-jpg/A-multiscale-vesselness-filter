import numpy as np
import Segmentation_filter as sf
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

list_of_results=[]
for t in range(3):
    for h in range (3):
        segmentated_image= np.array(sf.main_segmentation(b=0.2*(t+1) ,c=7*(h+1)))



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
b_values=np.zeros(10)
c_values=np.zeros(20)
for i in range(3):
      b_values[i]=(i+1)*0.2
      c_values[i]=(i+1)*7
      #c_values[i+10]=(i+11)*4


      

plt.plot(tau_values,list_of_results)
plt.xlabel("$\tau$ values")
plt.ylabel("AUPRC")
plt.show()
"""
# Source - https://stackoverflow.com/a
# Posted by Michael Szczesny, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-22, License - CC BY-SA 4.0
b_values=np.zeros(4)
c_values=np.zeros(4)
for i in range(4):
      b_values[i]=(i+1)*0.8
      c_values[i]= i+5
      #c_values[i+10]=(i+11)*4
def f(vector=[0.4,7], wn=0 , f="f"):
    v0=vector[0]
    v1=vector[1]
    segmentated_image= np.array(sf.main_segmentation(alpha=v0 ,beta=v1,wns=wn,filtre=f))


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
    for i in range(x):
                for j in range(y):
                    gris[i*j+j] =pixels[i,j]
    result=np.zeros((x,y))
    array_of_pixel_values=np.zeros((y,x,3))
    print(f"max du mask = {np.max(gris)}")
    print(np.max(gris))
    print(np.max(segmentated_image))
    si=np.zeros(x*y)
    for i in range(x):
                for j in range(y):
                    si[i*j+j] =segmentated_image[i,j]
    segmentated_image=(255/np.max(si))*(si.flatten())
    #print(f"vector = {vector}")
    #print(np.linalg.norm(segmentated_image-gris))
    """gris/=255
    segmentated_image/=255
    segmentated_image-=0.3
    """
    AUPRC=slm.average_precision_score(gris, segmentated_image, pos_label=255)
    return AUPRC
    
    print(f"AUPRC {AUPRC}")


x = b_values
y = c_values
z_flat = np.array([[f(vector=[i,j]) for i in x ]for j in y])
# 1. Create the grid
X, Y = np.meshgrid(x, y)
"""
# 2. Compute z in a flat list
z_flat1 = np.array([f(wn=t) for t in x ])
z_flat2 = np.array([f(wn=t , f="p") for t in x ])

# 3. Reshape z so it matches the meshgrid
Z = np.zeros((len(Y),len(X)))
for i in range(len(X)):
       for j in range(len(Y)):
              Z[j,i]=z_flat[j*i+i]
"""
Z =z_flat
# 4. Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('Frangi_s filter response as a function of it_s two parameters')

plt.show()
"""

plt.plot(x,z_flat1, label= "Frangi's response to noise")
plt.plot(x,z_flat2 , label ="Filtre proposé's response to noise")
plt.xlabel("noise scale")
plt.ylabel("AUPRC")
plt.legend()
plt.show()
"""

