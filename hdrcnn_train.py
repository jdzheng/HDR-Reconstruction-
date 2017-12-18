import os, sys
import tensorflow as tf
import tensorlayer as tl

from scipy import misc
from skimage import io
import numpy as np
import network  # , img_io

import glob
import imageio
from imgaug import augmenters as iaa
from PIL import Image

import random
import randomCrop
from datetime import datetime

'''
Input: Takes in a set of two of the same images -- one is the Low dynamic range version and the other is 
the high dynamic range version. 

Ouput: Either chooses to flip the image horizontally or not and then outputs it
'''
def getTransform(curr_LDR, curr_HDR):
	random.seed(datetime.now())
	randSeed = random.randint(1,1000000) # make sure both LDR and HDR are either flipped or not flipped
	flipping = lambda x: tf.image.random_flip_left_right(x, seed=randSeed) # use lambda to work with batches
	flipped_LDR = tf.map_fn(flipping, curr_LDR)
	flipped_HDR = tf.map_fn(flipping, curr_HDR)
	return flipped_LDR, flipped_HDR

# CONSTANTS
global_step = tf.get_variable('step', shape=(),
							  initializer=tf.constant_initializer(0),
							  regularizer=None,
							  trainable=False)
LEARNING_RATE = tf.train.exponential_decay(.00005, global_step, 1000, .98, staircase=False, name=None)
NUM_EPOCHS = 999999
DATA_SIZE = 4000
BATCH_SIZE = 1
LABELS = list(range(DATA_SIZE))
oldLoss = 999999
totalLoss = 0
pastLoss = 999999

# IMG-LDR
curr_LDR = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 320, 320, 3])
# IMG-HDR
curr_HDR = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 320, 320, 3])

y_input, y_actual = getTransform(curr_LDR, curr_HDR)
my_net = network.model(y_input)
y_pred = network.get_final(my_net, y_input)

# Loss functions
meanSquareLoss = tf.losses.mean_squared_error(labels=y_actual, predictions=y_pred)
absDevLoss = tf.losses.absolute_difference(labels=y_actual, predictions=y_pred)
huberLoss = tf.losses.huber_loss(labels=y_actual, predictions=y_pred)

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(huberLoss)
correct_prediction = tf.equal(tf.argmax(y_actual, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the saver to save the model
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# restores from your last run
	# if load_from_file is not None:
	#     saver.restore(sess, load_from_file)

	counter = 0
	# Run epochs
	for epoch in range(NUM_EPOCHS):
		print("=====================================================================")
		print("------------------------------", "EPOCH %d " % (epoch + 1), "--------------------------------")
		print("=====================================================================")

		# Run example on number of batches until we go through entire dataset
		numIterations = int(DATA_SIZE / BATCH_SIZE)

		copySet = LABELS[:] # To make a copy of the image indexes

		batchNum = 0 # Keep track of the batches

		for j in range(numIterations):

			# Initialize new arrays to keep track of LDR and HDR images
			new_LDR = []
			new_HDR = []

			# if BATCH_SIZE > len(copySet):
			# 	break

			curBatch = random.sample(copySet, BATCH_SIZE) # sample random set of images 

			batchNum+=1

			print((batchNum/numIterations)*100, "% Complete with Epoch")
			# Augments the data (both LDR and HDR) with a random crop and a horizontal flip 
			# and then appends the new images to the LDR and HDR arrays
			for index in curBatch:
				randSeed = datetime.now()
				LDR_path = "./Dataset/Input/LDR-" + str(index+1) + ".jpg"
				HDR_path = "./Dataset/Ground_Truth/HDR-" + str(index+1) + ".jpg"

				original_LDR = imageio.imread(LDR_path)
				original_HDR = imageio.imread(HDR_path)

				LDR_Aug = randomCrop.random_crop(original_LDR, randSeed)
				HDR_Aug = randomCrop.random_crop(original_HDR, randSeed)
				reScale = iaa.Scale(320)

				LDR_Aug = reScale.augment_image(LDR_Aug)
				HDR_Aug = reScale.augment_image(HDR_Aug)

				counter += 1
				new_LDR.append(LDR_Aug)
				new_HDR.append(HDR_Aug)

				# Removes already seen image from our set of all images
				print "Index to remove: ", index
				copySet.remove(index)

			# Saves progress after every 1000 iterations in seperate directory
			if counter % 1000 == 0:
				t, loss, prediction = sess.run([train_step, meanSquareLoss, y_pred],
										   feed_dict={curr_LDR: new_LDR, curr_HDR: new_HDR})
				imageio.imwrite('./metrics/LDR-Image-%d.jpg' % (counter + 1), new_LDR[0])
				imageio.imwrite('./metrics/Ground_Truth-%d.jpg' % (counter + 1), new_HDR[0])
				imageio.imwrite('./metrics/Prediction-%d.jpg' % (counter + 1), prediction[0])

				oldLoss = loss
			else:
				t, loss = sess.run([train_step, meanSquareLoss],
								   feed_dict={curr_LDR: new_LDR, curr_HDR: new_HDR})
				totalLoss+=loss

			# Displays total loss for ever 100 iterations	
			if counter % 100 == 0:
				totalLoss/=counter

				for i in range (3):
					print("************************************************************************************")
				print("************** Average Loss over past 100 Iterations: ", (totalLoss), "**************")
				print("************** % Decrease from Last 100: ", ((pastLoss-totalLoss)/pastLoss)*100, "% **************")
				for i in range(3):
					print("************************************************************************************")

				pastLoss = totalLoss
				totalLoss = 0
			print("~~~~~~~~~~~~~~~~~~~~~~ Loss: ", loss, "~~~~~~~~~~~~~~~~~~~~~~")
			print("------------------------- Step: ", counter, "-------------------------")

		# Save image to folder to see if training is progressing
		# Save the current training data
		if epoch % 9 == 0:
			saver.save(sess, './saved-models/my-model', global_step=epoch)

	sess.close()

