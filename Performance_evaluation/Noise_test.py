import numpy as np
#import Segmentation_filter as sf
#import MEDNR as md
import matplotlib.pyplot as plt
import numpy.random as rd
#import AUPRC_Estimation as AUPR

def add_white_noise(image, sc=1):
    y,x=image.shape
    wn = np.random.normal(scale=sc , size=(y,x))
    return wn+image
"""

for i in range(5):
    sf.main_segmentation(wns=0.5*i)

md_responsesf=[]
md_responsesp =[]
for i in range(5):
    segmented_image= np.array(sf.main_segmentation(wns=0.5*i)).flatten()
    md_responsesf.append(AUPR.(segmented_image))
    segmented_image_p= np.array(sf.main_segmentation(filtre ="p" , wns=0.5*i)).flatten()
    md_responsesp.append(md.MEDNR(segmented_image_p))
resultf=np.array(md_responsesf)
resultp=np.array(md_responsesp)
rap_change_cont=np.zeros(5)
for i in range(5):
    rap_change_cont[i]=0.2*i
plt.plot(rap_change_cont,resultf, label= "Frangi's response")
plt.plot(rap_change_cont,resultp , label ="Filtre proposé")
plt.xlabel("$\rapport de changement de contraste linéaire")
plt.ylabel("MEDNR")
plt.legend()
plt.show()
"""