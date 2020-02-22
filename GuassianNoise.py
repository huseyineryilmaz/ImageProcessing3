import cv2
import numpy as np
from PIL import Image, ImageDraw
import PIL

def show_image_and_wait(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise

    return noisy_image

def convert_to_uint8(image_in):
    temp_image = np.float64(np.copy(image_in))
    cv2.normalize(temp_image, temp_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)

    return temp_image.astype(np.uint8)

def medianFilter(input_image):
	img = input_image
	input_pixels = img.load()
	members = [(0,0)] * 9

	# Create output image
	output_image = Image.new("RGB", img.size)
	draw = ImageDraw.Draw(output_image)

	for i in range(1,img.width-1):
		for j in range(1,img.height-1):
			acc = [0, 0, 0]
			members[0] = img.getpixel((i-1,j-1))
			members[1] = img.getpixel((i-1,j))
			members[2] = img.getpixel((i-1,j+1))
			members[3] = img.getpixel((i,j-1))
			members[4] = img.getpixel((i,j))
			members[5] = img.getpixel((i,j+1))
			members[6] = img.getpixel((i+1,j-1))
			members[7] = img.getpixel((i+1,j))
			members[8] = img.getpixel((i+1,j+1))
			members.sort()
			pixel = members[4]
			acc[0] = pixel[0] 
			acc[1] = pixel[1] 
			acc[2] = pixel[2]
			draw.point((i,j),(int(acc[0]), int(acc[1]), int(acc[2])))
	output_image.save('MedianFilterForGuassianNoise.png')


def main():
    girl_face_filename = "resimOrjinal.png"
    print('opening image: ', girl_face_filename)
    girl_face_image = cv2.imread(girl_face_filename, cv2.IMREAD_UNCHANGED)
    girl_face_grayscale_image = girl_face_image 


    noisy_sigma = 35
    noisy_image = add_gaussian_noise(girl_face_grayscale_image, noisy_sigma)

    print('noisy image shape: {0}, len of shape {1}'.format(\
        girl_face_image.shape, len(noisy_image.shape)))
    print('    WxH: {0}x{1}'.format(noisy_image.shape[1], noisy_image.shape[0]))
    print('    image size: {0} bytes'.format(noisy_image.size))

    show_image_and_wait(girl_face_filename, convert_to_uint8(noisy_image))
    noisy_filename = 'GuassianNoise.png'
    cv2.imwrite(noisy_filename, noisy_image)
    
    img=Image.open('GuassianNoise.png')
    medianFilter(img)

if __name__ == "__main__":
    main()
