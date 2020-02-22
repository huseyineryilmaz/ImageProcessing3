import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageDraw
import PIL
import math
from math import sqrt
import random



def saltAndPeper(img):
	input_image = img
	input_pixels = input_image.load()

	# Create output image
	output_image = Image.new("RGB", input_image.size)
	draw = ImageDraw.Draw(output_image)

	#add salt and pepper noise
	for x in range(input_image.width):
		for y in range(input_image.height):
			acc = [0, 0, 0]
			pixel = input_pixels[x, y]
			rnd = random.randint(1,1000) #take a random integer between 1 and 1000 for every pixel 
			if(rnd>950):  #if random int >950 make pepper that pixel
				acc[0] = 0 
				acc[1] = 0 
				acc[2] = 0 
			elif(rnd<50):  #if random int <50 make salt that pixel
				acc[0] = 255 
				acc[1] = 255 
				acc[2] = 255
			else:		#if random int between 50 and 950 then do not any changes on pixel
				acc[0] = pixel[0] 
				acc[1] = pixel[1] 
				acc[2] = pixel[2] 
			draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))
    
	output_image.save('SaltAndPapper.png')


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
			members[0] = img.getpixel((i-1,j-1))  # It takes 9 pixels value
			members[1] = img.getpixel((i-1,j))
			members[2] = img.getpixel((i-1,j+1))
			members[3] = img.getpixel((i,j-1))
			members[4] = img.getpixel((i,j))
			members[5] = img.getpixel((i,j+1))
			members[6] = img.getpixel((i+1,j-1))
			members[7] = img.getpixel((i+1,j))
			members[8] = img.getpixel((i+1,j+1))
			members.sort()  	#sort the pixel values
			pixel = members[4] 	#replace the median pixel to other 9 pixels
			acc[0] = pixel[0] 
			acc[1] = pixel[1] 
			acc[2] = pixel[2]
			draw.point((i,j),(int(acc[0]), int(acc[1]), int(acc[2])))
	output_image.save('MedianFilterForSaltAndPepper.png')


def main():

	img=Image.open('resimOrjinal.png')
	saltAndPeper(img)
	img2 = Image.open('SaltAndPapper.png')
	medianFilter(img2)


if __name__ == "__main__":
    main()

