import os,sys
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt
#%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

#!diff imagenet/imagenet_full_conv.prototxt ../models/bvlc_reference_caffenet/deploy.prototxt

net = caffe.Net(caffe_root + 'models/BDGP_stage/deploy.prototxt',
                       caffe_root + 'models/BDGP_stage/BDGP_stage_iter_2000.caffemodel')
params = ['fc6', 'fc7', 'fc8_BDGP']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
#for fc in params:
#	print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

# random initialize the layers 4-8
# check the difference in python command line 
for fc in params:
	filters = net.params[fc][0].data
	print fc_params[fc][0].shape
	init = np.random.random(fc_params[fc][0].shape)
	fc_params[fc][0].data = init


net.save('../models/BDGP_stage/net_surgery3.caffemodel')
