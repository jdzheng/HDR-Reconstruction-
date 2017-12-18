from PIL import ImageEnhance
from PIL import Image
import numpy as np

def MSE(truth, prediction):
	meanSquareLoss = 0.0
	for index in range(len(truth)):
		diff = np.subtract(truth[index], prediction[index])
		squared = sum(diff)**2

		meanSquareLoss += float(squared)/len(truth)
	
	return meanSquareLoss

def basicHDR(image): 
	img = Image.open(image)

	#Bump contrast
	img = ImageEnhance.Contrast(img).enhance(1.0)
	#Bump saturation 
	img = ImageEnhance.Color(img).enhance(1.2)
	#Bump sharpness
	img = ImageEnhance.Sharpness(img).enhance(1.1)
	#Bump brightness
	img = ImageEnhance.Brightness(img).enhance(1.75)

	im = np.array(img)
	result = Image.fromarray(im)
	result.save('out.jpg')

if __name__ == "__main__":
	basicHDR("./Mean-Square-Predict/LDR-Image-7001.jpg")
	