from PIL import Image
import imageio
import numpy

def MSE(prediction, truth):
	pred = Image.open(prediction)
	im_pred = pred.load()

	truth = Image.open(truth)
	im_truth = truth.load()

	width, height = pred.size
	img_pred = [im_pred[i, j] for i in range(width) for j in range(height)]

	width, height = truth.size
	img_truth = [im_truth[i, j] for i in range(width) for j in range(height)]

	meanSquareLoss = 0.0
	for index in range(len(img_truth)):
		diff = numpy.subtract(img_truth[index], img_pred[index])
		squared = sum(diff)**2

		meanSquareLoss += float(squared)/len(img_truth)
	
	return meanSquareLoss

def imageToLuminaceValues(imageLDR, imageTruth, imagePred):
	imgLDR = Image.open(imageLDR)
	pixLDR = imgLDR.load()

	imgHDR = Image.open(imageTruth)
	pixHDR = imgHDR.load()

	imgPred = Image.open(imagePred)
	pixPred = imgPred.load()

	width, height = imgPred.size
	pixelsPrediction = [pixPred[i, j] for i in range(width) for j in range(height)]

	widthLDR, heightLDR = imgLDR.size
	pixelsLDR = [pixLDR[i, j] for i in range(widthLDR) for j in range(heightLDR)]

	widthHDR, heightHDR = imgHDR.size
	pixelsGroundTruth = [pixHDR[i, j] for i in range(width) for j in range(height)]

	brightnessValsPred = []
	brightnessValsHDR = []
	brightnessValsLDR= []

	for pixel in range(len(pixelsPrediction)):
		valsPred = pixelsPrediction[pixel]
		valsHDR = pixelsGroundTruth[pixel]
		valsLDR = pixelsLDR[pixel]

		brightnessValsLDR.append(0.299*valsLDR[0] + 0.587*valsLDR[1] + 0.114*valsLDR[2])
		brightnessValsHDR.append(0.299*valsHDR[0] + 0.587*valsHDR[1] + 0.114*valsHDR[2])
		brightnessValsPred.append(0.299*valsPred[0] + 0.587*valsPred[1] + 0.114*valsPred[2])


	print "Luminance ratio LDR: ", float(max(brightnessValsLDR))/(min(brightnessValsLDR)+0.25)
	print "Luminance ratio ground truth: ", float(max(brightnessValsHDR))/(min(brightnessValsHDR)+0.25)
	print "Luminance ratio prediction: ", float(max(brightnessValsPred))/(min(brightnessValsPred)+0.25)

def testLuminance():
	print "---------------------------------------------------------------------------"
	print "Huber Loss Using Baseline Prediction Luminance Ratio:"
	imageToLuminaceValues("./Huber-Loss-Predict/LDR-Image-69001.jpg", "./Huber-Loss-Predict/Ground_Truth-69001.jpg", 
		"./outHuber.jpg")
	print "---------------------------------------------------------------------------"
	print "Huber Loss Using Model Prediction Luminance Ratio :"
	imageToLuminaceValues("./Huber-Loss-Predict/LDR-Image-69001.jpg", "./Huber-Loss-Predict/Ground_Truth-69001.jpg", 
		"./Huber-Loss-Predict/Prediction-69001.jpg")
	print "---------------------------------------------------------------------------"
	print "Mean Square Loss Using Baseline Prediction Luminance Ratio :"
	imageToLuminaceValues("./Mean-Square-Predict/LDR-Image-7001.jpg", "./Mean-Square-Predict/Ground_Truth-7001.jpg", 
		"./out.jpg")
	print "---------------------------------------------------------------------------"
	print "Mean Square Loss Using Model Prediction Luminance Ratio :"
	imageToLuminaceValues("./Mean-Square-Predict/LDR-Image-7001.jpg", "./Mean-Square-Predict/Ground_Truth-7001.jpg", 
		"./Mean-Square-Predict/Prediction-119001.jpg")
	print "---------------------------------------------------------------------------"

def testMSE():
	print "---------------------------------------------------------------------------"
	print "MSE Huber Loss Using Baseline Prediction: ", MSE("./outHuber.jpg", "./Huber-Loss-Predict/Ground_Truth-69001.jpg")
	print "---------------------------------------------------------------------------"
	print "MSE Huber Loss Using Model Prediction: " , MSE("./Huber-Loss-Predict/Prediction-69001.jpg", "./Huber-Loss-Predict/Ground_Truth-69001.jpg")
	print "---------------------------------------------------------------------------"
	print "MSE Mean Square Loss Using Baseline Prediction: ", MSE("./out.jpg", "./Mean-Square-Predict/Ground_Truth-7001.jpg")
	print "---------------------------------------------------------------------------"
	print "Mean Square Loss Using Model Prediction: " , MSE("./Mean-Square-Predict/Prediction-119001.jpg", "./Mean-Square-Predict/Ground_Truth-7001.jpg")
	print "---------------------------------------------------------------------------"

if __name__ == "__main__":
	testLuminance()
	testMSE()





