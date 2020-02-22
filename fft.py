import cv2
import numpy as  np
from matplotlib import pyplot as plt

img = cv2.imread("resimOrjinal.png", cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(img)			#transform the image fft form
fshift = np.fft.fftshift(f)
magnitude_spectrum0 = 20*np.log(np.abs(fshift))
magnitude_spectrum0 = np.asarray(magnitude_spectrum0, dtype=np.uint8)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

rows, cols = img.shape
crow,ccol = rows/2 , cols/2

mask = np.zeros((rows,cols,2),np.uint8)
mask[int(crow-0):int(crow+750), int(ccol-0):int(ccol+750)] = 1  # we create a new fft mask(periodic noise)

fshift = dft_shift*mask #apply mask
f_ishift = np.fft.ifftshift(fshift)  #apply inverse fft, it use fft form to new image so it dont use the original picture, it takes the fft form and 
img_back = cv2.idft(f_ishift)		# convert it to image
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

mask2 = np.zeros((rows,cols,2),np.uint8)
mask2[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 1 

fshift2 = dft_shift*mask2 
f_ishift2 = np.fft.ifftshift(fshift2)  
img_back2 = cv2.idft(f_ishift2)		
img_back2 = cv2.magnitude(img_back2[:,:,0],img_back2[:,:,1])

mag_spectrum = [img, img_back,magnitude_spectrum0,img_back2]
filter_name = ['input image', 'Periodic Noise', 'FFT','Remove Periodic Noise']
for i in range(4):
    plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i],cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])

plt.show()

