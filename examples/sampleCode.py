import os,sys
import numpy as np
#import matplotlib.pyplot as plt
import pylab as plt
#matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

net = caffe.Classifier(caffe_root + 'examples/imagenet/alexnet_deploy.prototxt',
						caffe_root + 'examples/imagenet/caffe_alexnet_model')
                       #caffe_root + 'examples/imagenet/caffe_reference_imagenet_model')
net.set_phase_test()
net.set_mode_cpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
print [(k, v.data.shape) for k, v in net.blobs.items()]

imgLst = open('/home/qian/Desktop/ODU/FlyFISHInfo/imageList.txt', 'r').readlines()
n_Img = len(imgLst)

#feat2c = np.zeros((186624, n_Img))
#feat2n = np.zeros((n_Img, 186624))
feat2p = np.zeros((n_Img, 43264))
#feat3 = np.zeros((n_Img, 64896))
#feat4 = np.zeros((n_Img, 64896))
#feat5c = np.zeros((n_Img, 43264))
#feat5p = np.zeros((n_Img, 9216))
#feat6 = np.zeros((n_Img, 4096))	
#feat7 = np.zeros((n_Img, 4096))
#feat8 = np.zeros((n_Img, 1000))
for i in xrange(n_Img):
	if i % 10 == 0: print i
	scores = net.predict([caffe.io.load_image('/home/qian/Desktop/images_FISH/' + imgLst[i][:-1])])
	#tmp2c = net.blobs['conv2'].data[4]
	#feat2c[:, i] = tmp2c.reshape(186624)
	#tmp2n = net.blobs['norm2'].data[4]
	#feat2n[i, :] = tmp2n.reshape(1, 186624)
	tmp2p = net.blobs['pool2'].data[4]
	feat2p[i, :] = tmp2p.reshape(1, 43264)	
	#tmp3 = net.blobs['conv3'].data[4]
	#feat3[i, :] = tmp3.reshape(1, 64896)
	#tmp4 = net.blobs['conv4'].data[4]
	#feat4[i, :] = tmp4.reshape(1, 64896)
	#tmp5conv = net.blobs['conv5'].data[4]
	#feat5c[i, :] = tmp5conv.reshape(1, 43264)
	#tmp5pool = net.blobs['pool5'].data[4]
	#feat5p[i, :] = tmp5pool.reshape(1, 9216)
	#tmp6 = net.blobs['fc6'].data[4]
	#feat6[i, :] = tmp6.reshape(1, 4096)
	#tmp7 = net.blobs['fc7'].data[4]
	#feat7[i, :] = tmp7.reshape(1, 4096)
	#tmp8 = net.blobs['fc8'].data[4]
	#feat8[i, :] = tmp8.reshape(1, 1000)

np.savetxt('results_alex/Labeled_FlyFISH.txt', feat2p, fmt = '%-8.4f')
#np.savetxt('results_alex/feat2p.txt', feat2p, fmt = '%-8.4f')
#np.savetxt('results_alex/feat2n.txt', feat2n, fmt = '%-8.4f')
#np.savetxt('results_alex/feat5c.txt', feat5c, fmt = '%-8.4f')
#np.savetxt('results_alex/feat5p.txt', feat5p, fmt = '%-8.4f')
#np.savetxt('results_alex/feat6.txt', feat6, fmt = '%-8.4f')
#np.savetxt('results_alex/feat7.txt', feat7, fmt = '%-8.4f')
#np.savetxt('results_alex/feat8.txt', feat8, fmt = '%-8.4f')