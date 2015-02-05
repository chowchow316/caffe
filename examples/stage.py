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
    print "there"
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    return data

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_phase_test()
caffe.set_mode_cpu()

#net = caffe.Classifier(caffe_root + 'examples/imagenet/alexnet_deploy.prototxt',
#						caffe_root + 'examples/imagenet/caffe_alexnet_model')
net = caffe.Classifier(caffe_root + 'models/BDGP_stage/deploy.prototxt',
                       caffe_root + 'models/BDGP_stage/BDGP_stage_iter_2000.caffemodel')
                       #caffe_root + 'examples/imagenet/caffe_reference_imagenet_model')

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
# convert binaryproto to npy
a=caffe.io.caffe_pb2.BlobProto()
file=open('/home/shuo/Documents/caffe/data/BDGP_stage/BDGP_mean.binaryproto','rb')
data = file.read()
a.ParseFromString(data)
means=a.data
means=np.asarray(means)
means=means.reshape(3,256,256)
net.set_mean('data', means)  # ImageNet mean

net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
print [(k, v.data.shape) for k, v in net.blobs.items()]

imgLstr = open('/home/shuo/Documents/caffe/data/BDGP_stage/train.txt', 'r').readlines()
imgLstst = open('/home/shuo/Documents/caffe/data/BDGP_stage/test.txt', 'r').readlines()
n_Imgtr = len(imgLstr)
n_Imgtst = len(imgLstst) 
n_Img = n_Imgtr + n_Imgtst

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

for i in xrange(n_Imgtr):
	if i % 100 == 0: print i
	scores = net.predict([caffe.io.load_image(imgLstr[i][:-5].strip())])
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
	#tmp8 = net.blobs['fc8_BDGP'].data[4]
	#feat8[i, :] = tmp8.reshape(1, 1000)

for i in xrange(n_Imgtst):
	if i % 100 == 0: print i
	scores = net.predict([caffe.io.load_image(imgLstst[i][:-5].strip())])
	#tmp2c = net.blobs['conv2'].data[4]
	#feat2c[:, i] = tmp2c.reshape(186624)
	#tmp2n = net.blobs['norm2'].data[4]
	#feat2n[i, :] = tmp2n.reshape(1, 186624)
	tmp2p = net.blobs['pool2'].data[4]
	feat2p[i + n_Imgtr, :] = tmp2p.reshape(1, 43264)	
	#tmp3 = net.blobs['conv3'].data[4]
	#feat3[i + n_Imgtr, :] = tmp3.reshape(1, 64896)
	#tmp4 = net.blobs['conv4'].data[4]
	#feat4[i + n_Imgtr, :] = tmp4.reshape(1, 64896)
	#tmp5conv = net.blobs['conv5'].data[4]
	#feat5c[i + n_Imgtr, :] = tmp5conv.reshape(1, 43264)
	#tmp5pool = net.blobs['pool5'].data[4]
	#feat5p[i + n_Imgtr, :] = tmp5pool.reshape(1, 9216)
	#tmp6 = net.blobs['fc6'].data[4]
	#feat6[i + n_Imgtr, :] = tmp6.reshape(1, 4096)
	#tmp7 = net.blobs['fc7'].data[4]
	#feat7[i + n_Imgtr, :] = tmp7.reshape(1, 4096)
	#tmp8 = net.blobs['fc8_BDGP'].data[4]
	#feat8[i, :] = tmp8.reshape(1, 1000)

np.savetxt('/home/shuo/Documents/FlyExpress/feat_fine_tune/BG_2000_2p.txt', feat2p, fmt = '%-8.4f')
#np.savetxt('/home/shuo/Documents/FlyExpress/feat_fine_tune/BG_2000_3.txt', feat3, fmt = '%-8.4f')
#np.savetxt('/home/shuo/Documents/FlyExpress/feat_fine_tune/BG_2000_4.txt', feat4, fmt = '%-8.4f')
#np.savetxt('/home/shuo/Documents/FlyExpress/feat_fine_tune/BG_1000_5p.txt', feat5p, fmt = '%-8.4f')
#np.savetxt('/home/shuo/Documents/FlyExpress/feat_fine_tune/BG_1000_5c.txt', feat5c, fmt = '%-8.4f')
#np.savetxt('/home/shuo/Documents/FlyExpress/feat_fine_tune/BG_2000_6.txt', feat6, fmt = '%-8.4f')
#np.savetxt('/home/shuo/Documents/FlyExpress/feat_fine_tune/BG_2000_7.txt', feat7, fmt = '%-8.4f')
#np.savetxt('/home/shuo/Documents/FlyExpress/feat_fine_tune/BDGP_1000_8.txt', feat8, fmt = '%-8.4f')
