#!/usr/bin/python
# -*- coding:utf-8 -*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils
import shutil
from tqdm import tqdm
import os

"""
An implementation of the neural network used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""

class EDSR(object):

	def __init__(self, img_size=50, num_layers=32, feature_size=256, scale=2, output_channels=3):
		print("Building EDSR...")
		with tf.name_scope(name='image_input'):
			self.input = x = tf.placeholder(tf.float32, [None, img_size, img_size, output_channels])
			mean_x = tf.reduce_mean(x)
			image_input = x - mean_x
		with tf.name_scope(name='target_input'):
			self.target = y = tf.placeholder(tf.float32, [None, img_size*scale, img_size*scale, output_channels])
			mean_y = tf.reduce_mean(y)
			image_target = y - mean_x


		resnet_out = self.resnet(image_input, feature_size, num_layers, reuse=False, scope='_g')
		self.debug = self.resnet(image_target, feature_size, num_layers, reuse=True, scope='_g')
		g_output = self.upconv(resnet_out, scale, feature_size)

		self.g_ini_out = output = g_output  # slim.conv2d(x,output_channels,[3,3])
		self.g_ini_loss = g_ini_loss = tf.reduce_mean(tf.losses.absolute_difference(image_target, g_output))  # L1 loss

		mse = tf.reduce_mean(tf.squared_difference(image_target, g_output))	
		self.PSNR = PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(tf.constant(255**2,dtype=tf.float32)/mse)
	

		# discriminator
		d_resnet_fake = self.resnet(g_output, feature_size, num_layers, reuse=True, scope='_g')
		d_logits_fake = self.classification(d_resnet_fake, feature_size, reuse=False)
		# d_resnet_real.shape = (batchsize, 100, 100, 128)
		self.d_resnet_real = d_resnet_real = self.resnet(image_target, feature_size, num_layers, reuse=True, scope='_g')
		self.d_logits_real = d_logits_real = self.classification(d_resnet_real, feature_size, reuse=True)
		
		d_loss1 = tf.losses.sigmoid_cross_entropy(d_logits_real, tf.ones_like(d_logits_real), scope='d1')
		d_loss2 = tf.losses.sigmoid_cross_entropy(d_logits_fake, tf.zeros_like(d_logits_fake), scope='d2')
		
		self.d_loss = d_loss1 + d_loss2

		# generator
		g_res_loss = 2e-6 * tf.reduce_mean(tf.squared_difference(d_resnet_real, d_resnet_fake))
		g_gan_loss = 1e-3 * tf.losses.sigmoid_cross_entropy(d_logits_fake, tf.ones_like(d_logits_fake), scope='g')
		self.g_loss = g_ini_loss + g_gan_loss + g_res_loss

		self.g_ini_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'resnet_g')+tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'upconv')
		self.d_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'resnet_g')+tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'classification')
		self.g_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'resnet_g')+tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'upconv')

		tf.summary.scalar("loss",self.g_ini_loss)
		tf.summary.scalar("PSNR",PSNR)
		tf.summary.image("input_image",self.input+mean_x)
		tf.summary.image("target_image",self.target+mean_y)
		tf.summary.image("output_image",self.g_ini_out+mean_x)
		
		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		print("Done building!")


	def resnet(self, image_input, feature_size, num_layers, scaling_factor=0.1, reuse=False, scope=''):
		with tf.variable_scope("resnet"+scope, reuse=reuse) as vs:
			x = slim.conv2d(image_input, feature_size, [3,3])
		
			conv_1 = x	
			
			for i in range(num_layers):
				x = utils.resBlock(x, feature_size, scale=scaling_factor)
			x = slim.conv2d(x, feature_size, [3,3])
			x += conv_1
		
		return x


	def upconv(self, x, scale, feature_size):
		#Upsample output of the convolution		
		with tf.variable_scope('upconv', reuse=False) as vs:
			x = utils.upsample(x, scale, feature_size, None)
		return x

	def classification(self, image_input, feature_size, reuse=False):
		with tf.variable_scope('calssification', reuse=reuse):
			x = slim.max_pool2d(image_input, [2, 2])
			x = slim.conv2d(x, feature_size*2, [3, 3])
			x = slim.max_pool2d(x, [2, 2])
			x = slim.conv2d(x, 1, [3, 3])
			self.debug = x
			x = tf.reduce_mean(x, axis=[1, 2])
			x = tf.nn.softmax(x)
		return x

	"""
	Save the current state of the network to file
	"""
	def save(self, savedir='saved_models'):
		print("Saving...")
		self.saver.save(self.sess, os.path.join(savedir, "model"))
		print("Saved!")
		
	"""
	Resume network from previously saved weights
	"""
	def resume(self,savedir='saved_models'):
		print("Restoring...")
		self.saver.restore(self.sess,tf.train.latest_checkpoint(savedir))
		print("Restored!")	


	def set_data_fn(self, fn, args, test_set_fn=None, test_set_args=None):
		self.data = fn
		self.args = args
		self.test_data = test_set_fn
		self.test_args = test_set_args

	"""
	Train the neural network
	"""
	def train(self, iterations=1000, save_dir="saved_models"):
		#Removing previous save directory if there is one
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)
		#Make new save directory
		os.mkdir(save_dir)
		#Just a tf thing, to merge all summaries into one
		merged = tf.summary.merge_all()

		train_op_g_ini = tf.train.AdamOptimizer().minimize(self.g_ini_loss, var_list=self.g_ini_var_list)
		train_op_d = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=self.d_var_list)
		train_op_g = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=self.g_var_list)

		#Operation to initialize all variables
		init = tf.global_variables_initializer()
		print("Begin training...")
		with self.sess as sess:
			#Initialize all variables
			sess.run(init)
			test_exists = self.test_data
			#create summary writer for train
			train_writer = tf.summary.FileWriter(save_dir+"/train",sess.graph)

			#If we're using a test set, include another summary writer for that
			if test_exists:
				test_writer = tf.summary.FileWriter(save_dir+"/test",sess.graph)
				test_x,test_y = self.test_data(*self.test_args)
				test_feed = {self.input:test_x,self.target:test_y}

			#This is our training loop
			for i in range(iterations):
				#Use the data function we were passed to get a batch every iteration
				x, y = self.data(*self.args)
				#Create feed dictionary for the batch
				feed = {self.input: x, self.target: y}
				#Run the train op and calculate the train summary
				summary, _, loss = sess.run([merged, train_op_g_ini, self.g_ini_loss], feed)
				print(loss)
				#If we're testing, don't train on test set. But do calculate summary
				if test_exists:
					t_summary = sess.run(merged,test_feed)
					#Write test summary
					test_writer.add_summary(t_summary,i)
				#Write train summary for this step
				train_writer.add_summary(summary,i)


			for i in range(iterations*10):
				#Use the data function we were passed to get a batch every iteration
				x, y = self.data(*self.args)
				#Create feed dictionary for the batch
				feed = {self.input: x, self.target: y}
				#Run the train op and calculate the train summary
				summary, _, d_loss_evl, test = sess.run([merged,train_op_d, self.d_loss, self.debug],feed)
				summary, _, g_loss_evl = sess.run([merged,train_op_g, self.g_loss],feed)

				print(test.shape)
				# print('d_loss:%f  g_loss:%f'%(d_loss_evl, g_loss_evl))
				#If we're testing, don't train on test set. But do calculate summary
				if test_exists:
					t_summary = sess.run(merged,test_feed)
					#Write test summary
					test_writer.add_summary(t_summary,i+iterations)
				#Write train summary for this step
				train_writer.add_summary(summary,i+iterations)
			# Save our trained model		
			self.save()	


	def predict(self,x):
		print("Predicting...")
		return self.sess.run([self.g_ini_out, self.PSNR],feed_dict={self.input:x})