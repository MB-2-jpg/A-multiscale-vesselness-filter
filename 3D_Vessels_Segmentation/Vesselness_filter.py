"""
TODO :

-passer en niveau de gris
-augmenter le contrast avec la methode de l'histograme ?
-
"""

from PIL import Image
import numpy as np
from scipy import ndimage

# Ouvrir
#img = Image.open("01_test.tiff")
def main_segmentation(image_name="01_test.tif", input_as_array=False ,pixels_array=None):
    if input_as_array == False :

        img = Image.open(image_name)

        # Dimensions
        #print(img.size, img.mode)

        # Lire pixels
        pixels = img.load()


        pixels_original = img.copy().load()
        #ACP= img.load()
        x, y = img.size


        #Niveau de gris
        gris = np.zeros((x,y))
        result=np.zeros((x,y))
        array_of_pixel_values=np.zeros((y,x,3))
        def calcul_gris():
            
            for i in range(x):
                for j in range(y):
                    r, g, b = pixels[i, j]
                    n = int((r+g+b)/3)
                    array_of_pixel_values[j][i] = np.array([r,g,b])
                    gris[i,j] = n
                    #if n>0:
                        #print(f'gris à la position (i,j)={gris[i,j]}')
        calcul_gris()
        print(f'gris={gris}')
        def calcul_gris_inverse():
            for i in range(x):
                for j in range(y):
                    r, g, b = pixels[i, j]
                    n = 255-int((r+g+b)/3)
                    pixels[i, j] = (n, n, n)
                    gris[i,j] = n

        #calcul_gris_inverse()
        def calcul_vecteur_empirique_d_analyse_en_composante_principale(im_gris_segmentée_prééliminaire,pixels,seuil=0.1):
            #print(f'image_gris_ségmentée_préeliminaire = {im_gris_segmentée_prééliminaire}')
            
            valeurs_après_ségmentation=im_gris_segmentée_prééliminaire.flatten()
            #print(f'valeurs_après_ségmentation = {valeurs_après_ségmentation}')
            minimal_value=min(valeurs_après_ségmentation)
            maximal_value=max(valeurs_après_ségmentation)
            treshhold= seuil*(maximal_value-minimal_value)+ minimal_value
            n=len(im_gris_segmentée_prééliminaire)
            p=len(im_gris_segmentée_prééliminaire[0])
            in_V_indices_list=[]
            not_in_V_indices_list=[]
            for i in range(n):
                if valeurs_après_ségmentation[i]>treshhold:
                    in_V_indices_list.append(i)
                    print(f'{i} in V')
                else:
                    not_in_V_indices_list.append(i)         
            v=np.zeros((1,3))
            pixels_values_flattened=(array_of_pixel_values).flatten()
            #print(f'pixels values _flattened ={pixels_values_flattened}') 
            for i in in_V_indices_list:
                for j in not_in_V_indices_list:
                    pixels_difference= pixels_values_flattened[i]-pixels_values_flattened[j]
                    v+=np.transpose(pixels_difference)
            norm_of_v=np.linalg.norm(v)
            if norm_of_v !=0:
                v/=norm_of_v
            print(v)
            return v
        def calcul_composante_principale_segmentation(v,im,x,y):
            
            ACP_seg=np.zeros((x,y))
            for i in range(x):
                for j in range(y):
                    

                    n = v@np.array(array_of_pixel_values[j][i])
                # pixels[i, j] = (n, n, n)
                    ACP_seg[i][j] = n
                    #print(f'ACP_seg[{i}][{j}]={n}')
            return ACP_seg
                





        def convolution(s):
            h11 = ndimage.gaussian_filter(gris, sigma=(s,0.7*s), order=(2, 0))
            h22 = ndimage.gaussian_filter(gris, sigma=(s,0.7*s), order=(0, 2))
            h12 = ndimage.gaussian_filter(gris, sigma=(s,0.7*s), order=(1, 1))
            return [h11, h12, h12, h22]

        def hessian_eigenvalues_2d(H):
            
            lambda1 = np.zeros((x,y))
            lambda2 = np.zeros((x,y))
            
            for i in range(x):
                for j in range(y):
                    h = np.zeros((2,2))
                    h[0,0] = H[0][i,j]
                    h[0,1] = H[1][i,j]
                    h[1,0] = H[2][i,j]
                    h[1,1] = H[3][i,j]

                    tr = np.trace(h)
                    det = np.linalg.det(h)
                    temp = np.sqrt(tr**2 - 4*det)
                    l1 = (tr + temp) / 2
                    l2 = (tr - temp) / 2

                    if max(abs(l1), abs(l2)) == abs(l1) :
                        tempo = l1
                        l1 = l2
                        l2 = tempo

                    lambda1[i,j] = l1
                    lambda2[i,j] = l2
            
            return lambda1, lambda2

            


        """
        b (sensibilité à la forme tubulaire via R)
        Typiquement entre 0.5 et 1.0.
        Petit b → favorise les structures très allongées (fort contraste tube vs fond).
        Grand b → accepte aussi des structures moins tubulaires (plus tolérant).

        c (sensibilité au contraste via S)
        Dépend du bruit et du niveau de contraste des vaisseaux.
        Si tes vaisseaux sont très contrastés : prends c plus grand (par ex. 200–500 selon l’échelle des intensités).
        Si tes vaisseaux sont fins et peu visibles : prends c plus petit (50–100).
        rq : je prefere des valeurs entre 1 et 3 mais je ne sais pas si c'est normal

        """

        def frangi(l1, l2):
            if (l2 < 0) : 
                return 0
            else : 
                b = 0.5
                c = 3.0
                r = abs(l1/l2) #Si r petit, alors espace courbe que dans une dimension, alors tube et pas sphere. Donc avec exp(-r) on ne garde que les tubes
                s = np.sqrt(l1*l1 + l2*l2)
                couleur = int(np.nan_to_num(255*np.exp(-r*r/(2*b*b))*(1-np.exp(-s*s/(2*c*c))), nan=0.0))                
                return couleur

        def calcul_lp(l2, l2_max, tau):
            if l2 > l2_max*tau :
                return l2
            else : 
                return l2_max*tau


        #t appartient a [0,1] est un seuil. Je ne crois pas qu'il soit dépandant de s. Voir comment le choisir.
        def papier(l2, l2_max, tau):
            lp = calcul_lp(l2, l2_max, tau)
            if l2 < 0 :
                return 0
            elif l2 >= lp/2:  
                return 1
            else :
                return l2*l2*(lp-l2)/((l2+lp)*(l2+lp))
        #Je pense que normaliser le résultat du filtre de frangi peut permettre une identification visuelle plus simple des vaisseaux

        # Hessienne avec convolution gaussienne
        s_min = 0.2
        s_max = 3
        s_delta = 0.5
        tau = 1
        calcul_gris()

        for s in np.arange(s_min, s_max, s_delta) : 
            print(s)
            h = convolution(s)
            lambda1, lambda2 = hessian_eigenvalues_2d(h)
            l2_max = np.max(lambda2) #max et pas min car je travail plutot avec des valeur positives
            for i in range(x):
                for j in range(y):
                    l1, l2 = lambda1[i,j], lambda2[i,j]
                    v = frangi(l1, l2)
                    #v = papier(l2, l2_max, tau)
                    result[i, j] = max(v, result[i,j])
                    c=int(result[i, j])
                    pixels[i, j] = (c, c, c)


        #img.save("filtre_frangi_nouvelle_gaussiène.png")


        v2=calcul_vecteur_empirique_d_analyse_en_composante_principale(result,pixels,seuil=0.2)
        composante_principale_pour_la_segmentation=calcul_composante_principale_segmentation(v2,pixels,x,y)
        for i in range(x):
            for j in range(y):
                c = int(composante_principale_pour_la_segmentation[i,j])
                pixels[i, j] = (c, c, c)
        img.save("ACP.png")


        def convolution2(s):
            h11 = ndimage.gaussian_filter(composante_principale_pour_la_segmentation, sigma=(s,0.7*s), order=(2, 0))
            h22 = ndimage.gaussian_filter(composante_principale_pour_la_segmentation, sigma=(s,0.7*s), order=(0, 2))
            h12 = ndimage.gaussian_filter(composante_principale_pour_la_segmentation, sigma=(s,0.7*s), order=(1, 1))
            return [h11, h12, h12, h22]


        for s in np.arange(s_min, s_max, s_delta) : 
            print(s)
            h = convolution2(s)
            lambda1, lambda2 = hessian_eigenvalues_2d(h)
            l2_max = np.max(lambda2) #max et pas min car je travail plutot avec des valeur positives
            for i in range(x):
                for j in range(y):
                    l1, l2 = lambda1[i,j], lambda2[i,j]
                    v = frangi(l1, l2)
                    #v = papier(l2, l2_max, tau)
                    result[i, j] = max(v, result[i,j])
                    c=int(result[i, j])
                    pixels[i, j] = (c, c, c)

        img.save("filtre_frangi_nouvelle_gaussiène_après_transformation_linéaire.png")





    else:
        pixels = pixels_array


        pixels_original = pixels_array
        #ACP= img.load()
        x, y = pixels.size


        #Niveau de gris
        gris = np.zeros((x,y))
        result=np.zeros((x,y))
        array_of_pixel_values=np.zeros((y,x,3))
        def calcul_gris():
            
            for i in range(x):
                for j in range(y):
                    r, g, b = pixels[i, j]
                    n = int((r+g+b)/3)
                    array_of_pixel_values[j][i] = np.array([r,g,b])
                    gris[i,j] = n
                    #if n>0:
                        #print(f'gris à la position (i,j)={gris[i,j]}')
        calcul_gris()
        print(f'gris={gris}')
        def calcul_gris_inverse():
            for i in range(x):
                for j in range(y):
                    r, g, b = pixels[i, j]
                    n = 255-int((r+g+b)/3)
                    pixels[i, j] = (n, n, n)
                    gris[i,j] = n

        #calcul_gris_inverse()
        def calcul_vecteur_empirique_d_analyse_en_composante_principale(im_gris_segmentée_prééliminaire,pixels,seuil=0.1):
            #print(f'image_gris_ségmentée_préeliminaire = {im_gris_segmentée_prééliminaire}')
            
            valeurs_après_ségmentation=im_gris_segmentée_prééliminaire.flatten()
            #print(f'valeurs_après_ségmentation = {valeurs_après_ségmentation}')
            minimal_value=min(valeurs_après_ségmentation)
            maximal_value=max(valeurs_après_ségmentation)
            treshhold= seuil*(maximal_value-minimal_value)+ minimal_value
            n=len(im_gris_segmentée_prééliminaire)
            p=len(im_gris_segmentée_prééliminaire[0])
            in_V_indices_list=[]
            not_in_V_indices_list=[]
            for i in range(n):
                if valeurs_après_ségmentation[i]>treshhold:
                    in_V_indices_list.append(i)
                    print(f'{i} in V')
                else:
                    not_in_V_indices_list.append(i)         
            v=np.zeros((1,3))
            pixels_values_flattened=(array_of_pixel_values).flatten()
            #print(f'pixels values _flattened ={pixels_values_flattened}') 
            for i in in_V_indices_list:
                for j in not_in_V_indices_list:
                    pixels_difference= pixels_values_flattened[i]-pixels_values_flattened[j]
                    v+=np.transpose(pixels_difference)
            norm_of_v=np.linalg.norm(v)
            if norm_of_v !=0:
                v/=norm_of_v
            print(v)
            return v
        def calcul_composante_principale_segmentation(v,im,x,y):
            
            ACP_seg=np.zeros((x,y))
            for i in range(x):
                for j in range(y):
                    

                    n = v@np.array(array_of_pixel_values[j][i])
                # pixels[i, j] = (n, n, n)
                    ACP_seg[i][j] = n
                    #print(f'ACP_seg[{i}][{j}]={n}')
            return ACP_seg
                





        def convolution(s):
            h11 = ndimage.gaussian_filter(gris, sigma=(s,0.7*s), order=(2, 0))
            h22 = ndimage.gaussian_filter(gris, sigma=(s,0.7*s), order=(0, 2))
            h12 = ndimage.gaussian_filter(gris, sigma=(s,0.7*s), order=(1, 1))
            return [h11, h12, h12, h22]

        def hessian_eigenvalues_2d(H):
            
            lambda1 = np.zeros((x,y))
            lambda2 = np.zeros((x,y))
            
            for i in range(x):
                for j in range(y):
                    h = np.zeros((2,2))
                    h[0,0] = H[0][i,j]
                    h[0,1] = H[1][i,j]
                    h[1,0] = H[2][i,j]
                    h[1,1] = H[3][i,j]

                    tr = np.trace(h)
                    det = np.linalg.det(h)
                    temp = np.sqrt(tr**2 - 4*det)
                    l1 = (tr + temp) / 2
                    l2 = (tr - temp) / 2

                    if max(abs(l1), abs(l2)) == abs(l1) :
                        tempo = l1
                        l1 = l2
                        l2 = tempo

                    lambda1[i,j] = l1
                    lambda2[i,j] = l2
            
            return lambda1, lambda2

            


        """
        b (sensibilité à la forme tubulaire via R)
        Typiquement entre 0.5 et 1.0.
        Petit b → favorise les structures très allongées (fort contraste tube vs fond).
        Grand b → accepte aussi des structures moins tubulaires (plus tolérant).

        c (sensibilité au contraste via S)
        Dépend du bruit et du niveau de contraste des vaisseaux.
        Si tes vaisseaux sont très contrastés : prends c plus grand (par ex. 200–500 selon l’échelle des intensités).
        Si tes vaisseaux sont fins et peu visibles : prends c plus petit (50–100).
        rq : je prefere des valeurs entre 1 et 3 mais je ne sais pas si c'est normal

        """

        def frangi(l1, l2):
            if (l2 < 0) : 
                return 0
            else : 
                b = 0.5
                c = 3.0
                r = abs(l1/l2) #Si r petit, alors espace courbe que dans une dimension, alors tube et pas sphere. Donc avec exp(-r) on ne garde que les tubes
                s = np.sqrt(l1*l1 + l2*l2)
                couleur = int(np.nan_to_num(255*np.exp(-r*r/(2*b*b))*(1-np.exp(-s*s/(2*c*c))), nan=0.0))                
                return couleur

        def calcul_lp(l2, l2_max, tau):
            if l2 > l2_max*tau :
                return l2
            else : 
                return l2_max*tau


        #t appartient a [0,1] est un seuil. Je ne crois pas qu'il soit dépandant de s. Voir comment le choisir.
        def papier(l2, l2_max, tau):
            lp = calcul_lp(l2, l2_max, tau)
            if l2 < 0 :
                return 0
            elif l2 >= lp/2:  
                return 1
            else :
                return l2*l2*(lp-l2)/((l2+lp)*(l2+lp))
        #Je pense que normaliser le résultat du filtre de frangi peut permettre une identification visuelle plus simple des vaisseaux

        # Hessienne avec convolution gaussienne
        s_min = 0.2
        s_max = 3
        s_delta = 0.5
        tau = 1
        calcul_gris()

        for s in np.arange(s_min, s_max, s_delta) : 
            print(s)
            h = convolution(s)
            lambda1, lambda2 = hessian_eigenvalues_2d(h)
            l2_max = np.max(lambda2) #max et pas min car je travail plutot avec des valeur positives
            for i in range(x):
                for j in range(y):
                    l1, l2 = lambda1[i,j], lambda2[i,j]
                    v = frangi(l1, l2)
                    #v = papier(l2, l2_max, tau)
                    result[i, j] = max(v, result[i,j])
                    c=int(result[i, j])
                    pixels[i, j] = (c, c, c)


        #img.save("filtre_frangi_nouvelle_gaussiène.png")


        v2=calcul_vecteur_empirique_d_analyse_en_composante_principale(result,pixels,seuil=0.2)
        composante_principale_pour_la_segmentation=calcul_composante_principale_segmentation(v2,pixels,x,y)
        for i in range(x):
            for j in range(y):
                c = int(composante_principale_pour_la_segmentation[i,j])
                pixels[i, j] = (c, c, c)
        img.save("ACP.png")


        def convolution2(s):
            h11 = ndimage.gaussian_filter(composante_principale_pour_la_segmentation, sigma=(s,0.7*s), order=(2, 0))
            h22 = ndimage.gaussian_filter(composante_principale_pour_la_segmentation, sigma=(s,0.7*s), order=(0, 2))
            h12 = ndimage.gaussian_filter(composante_principale_pour_la_segmentation, sigma=(s,0.7*s), order=(1, 1))
            return [h11, h12, h12, h22]


        for s in np.arange(s_min, s_max, s_delta) : 
            print(s)
            h = convolution2(s)
            lambda1, lambda2 = hessian_eigenvalues_2d(h)
            l2_max = np.max(lambda2) #max et pas min car je travail plutot avec des valeur positives
            for i in range(x):
                for j in range(y):
                    l1, l2 = lambda1[i,j], lambda2[i,j]
                    v = frangi(l1, l2)
                    #v = papier(l2, l2_max, tau)
                    result[i, j] = max(v, result[i,j])
                    c=int(result[i, j])
                    pixels[i, j] = (c, c, c)

        return pixels


"""


result=np.fft.fft2(gris)
for i in range(x):
    for j in range(y):
        c = int(abs(result[i,j]))
        pixels[i, j] = (c, c, c)
img.save("tftd2d.png")
"""