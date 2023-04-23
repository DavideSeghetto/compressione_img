import cv2
import numpy as np
import matplotlib.pyplot as plt

# Calcolo dell'MSE tra componente originale e compressa
def mse(cmp1, cmp2):
    return np.mean((cmp1 - cmp2) ** 2)

# Calcolo dell'MSE pesato
def weighted_mse(mse_list):
    return 3/4 * mse_list[0] + 1/8 * mse_list[1] + 1/8 * mse_list[2]

# Calcolo del PSNR pesato
def weighted_psnr(mse_p):
    return 10 * np.log10((255 ** 2) / mse_p)

#Funzione che effettua la "compressione" dell'immagine
def cmp_img(img, N, R):
    # Converto l'immagine in YCrCb (quando cv2 importa un immagine essa è in formato BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    #Estraggo le tre componenti e le salvo in una lista
    components = list(cv2.split(img))

    #List in cui salvo i vari valori dell'mse
    mse_list = []

    #Matrice tridimensionale che conterrà i valori "compressi" e che rappresenterà la nuova immagine "compressa"
    res = np.float64(np.zeros(img.shape))

    for k in range(0, 3):
        #Matrice bidimensionale che conterrà i valori ottenuti dalla dct
        tmp = np.float64(np.zeros(components[k].shape))

        for i in range(0, components[k].shape[0], N):
            for j in range(0, components[k].shape[1], N):
                # Estraggo il blocco NxN dalla componente
                block = components[k][i : i + N, j : j + N]

                # Applico la dct al blocco
                block = cv2.dct(np.float64(block))

                # Assegnamento del blocco "trasformato" alla matrice temporanea
                tmp[i: i + N, j: j + N] = block

        # Calcolo la soglia di azzeramento
        threshold = np.percentile(np.abs(tmp), R)

        # Azzeramento dei valori sotto la soglia
        tmp[np.abs(tmp) < threshold] = 0

        for i in range(0, components[k].shape[0], N):
            for j in range(0, components[k].shape[1], N):
                # Estraggo il blocco NxN dalla matrice di valori ottenuti dalla dct
                block = tmp[i : i + N, j : j + N]

                # Applico la dct inversa al blocco
                block = np.abs(cv2.idct(np.float64(block)))

                # Nel caso in cui la dct mi abbia restituito dei valori assoluti maggiori di 255 setto tali pixel a 255
                block[block > 255] = 255

                # Assegnamento del blocco "compresso" alla matrice tridimensionale
                res[i: i + N, j: j + N, k] = block

        #Calcolo l'mse tra componente originale e compressa e aggiungo il risultato alla lista
        mse_list.append(mse(components[k], res[:,:,k]))

    # Conversione a BGR
    res = cv2.cvtColor(np.uint8(res), cv2.COLOR_YCrCb2BGR)

    return res, mse_list

#Importo l'immagine
img = cv2.imread("colors.bmp")

#Lista contente i valori del PSNR pesato
psnr_list = []

#Range dei valori di R e N
Rs = range(10, 110, 10)
Ns = [8, 16, 64]

#for N in Ns:
for R in Rs:
    # Comprimo l'immagine
    compressed_img, mse_list = cmp_img(img, 8, R)

    #Calcolo dell'MSE pesato
    mse_w = weighted_mse(mse_list)
    print("L'MSE pesato per R = " + str(R) + " è: " + str(mse_w))

    #Calcolo del PSNR pesato e aggiungo il valore appena calcolato alla lista
    psnr_w = weighted_psnr(mse_w)
    psnr_list.append(psnr_w)
    print("Il PSNR per R = " + str(R) + " è: " + str(psnr_w))
    print()
    print()

# Traccio il grafico del PSNR in funzione di R
plt.plot(Rs, psnr_list)
plt.title("Curva del PSNR in funzione di R")
plt.xlabel("R")
plt.ylabel(" PSNR")
plt.xlim(10, 100)
plt.grid()
plt.show()

# Esempi prof
compressed_img, mse_list = cmp_img(img, 8, 97)
mse_w = weighted_mse(mse_list)
print("L'MSE pesato è: " + str(mse_w))
psnr_w = weighted_psnr(mse_w)
psnr_list.append(psnr_w)
print("Il psnr è: " + str(psnr_w))
print()
print()
cv2.imshow("N_8_R_97", compressed_img)

compressed_img, mse_list = cmp_img(img, 16, 99)
mse_w = weighted_mse(mse_list)
print("L'MSE pesato è: " + str(mse_w))
psnr_w = weighted_psnr(mse_w)
psnr_list.append(psnr_w)
print("Il psnr è: " + str(psnr_w))
print()
print()
cv2.imshow("N_16_R_99", compressed_img)

cv2.waitKey(0)
cv2.destroyAllWindows()