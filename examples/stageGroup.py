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

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    return data

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


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

imgLst = open('/home/qian/Desktop/stageInfo/imgS4.txt', 'r').readlines()
n_Img = len(imgLst)

#feat2c = np.zeros((186624, n_Img))
#feat2n = np.zeros((n_Img, 186624))
#feat2p = np.zeros((n_Img, 43264))
#feat3 = np.zeros((n_Img, 384))
feat4 = np.zeros((n_Img, 384))
#feat5c = np.zeros((n_Img, 43264))
#feat5p = np.zeros((n_Img, 9216))
#feat6 = np.zeros((n_Img, 4096))	
#feat7 = np.zeros((n_Img, 4096))
#feat8 = np.zeros((n_Img, 1000))
for i in xrange(n_Img):
	if i % 10 == 0: print i
	scores = net.predict([caffe.io.load_image('/home/qian/Desktop/BDGPimages/' + imgLst[i][:-1])])
	#tmp3 = net.blobs['conv3'].data[4]
	tmp4 = net.blobs['conv4'].data[4]
	#print tmp3.shape
	#print tmp3.shape[0]
	#print tmp3[0, :, :].shape
	#print tmp3[0, :, :].mean()
	for j in xrange(tmp4.shape[0]):
		#feat3[i, j] = tmp3[j, :, :].mean()
		feat4[i, j] = tmp4[j, :, :].mean() 


#np.savetxt('results_alex/feat3Group.txt', feat3, fmt = '%-8.4f')
np.savetxt('results_alex/feat4Group.txt', feat4, fmt = '%-8.4f')
