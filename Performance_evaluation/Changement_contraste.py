"""import numpy as np
import Segmentation_filter as sf
import MEDNR as md
import matplotlib.pyplot as plt

segmented_image= np.array(sf.main_segmentation()).flatten()
M=np.max(segmented_image)

segmented_image = segmented_image*(log(255)/M)

exp=np.exp((segmented_image))
racine= (segmented_image*(255**2/M))**0.5

md_responsesf=[]
md_responsesp =[]
for i in range(5):
    segmented_image= np.array(sf.main_segmentation(change_contraste=0.2*i)).flatten()
    md_responsesf.append(md.MEDNR(segmented_image))
    segmented_image_p= np.array(sf.main_segmentation(filtre ="p" , change_contraste=0.2*i)).flatten()
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