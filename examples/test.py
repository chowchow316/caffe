import numpy as np
#import matplotlib.pyplot as plt
import pylab as plt
#matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net = caffe.Classifier(caffe_root + 'examples/imagenet/imagenet_deploy.prototxt',
                       caffe_root + 'examples/imagenet/caffe_reference_imagenet_model')
net.set_phase_test()
net.set_mode_cpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

scores = net.predict([caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')])

#print scores
print [(k, v.data.shape) for k, v in net.blobs.items()]


plt.clf()
plt.figure()
plt.imshow(net.deprocess('data', net.blobs['data'].data[4]))
#plt.show()
plt.savefig('original.png')

# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
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

# index four is the center crop
plt.imshow(net.deprocess('data', net.blobs['data'].data[4]))
plt.savefig('2.png')

# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
plt.clf()
plt.figure()
plt.imshow(vis_square(filters.transpose(0, 2, 3, 1)))
plt.savefig('conv1_para.png')

#The first layer output, conv1
feat = net.blobs['conv1'].data[4, :36]
plt.clf()
plt.figure()
plt.imshow(vis_square(feat, padval=1))
plt.savefig('conv1_feat.png')

# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
#try:
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
#except:
#    ../data/ilsvrc12/get_ilsvrc_aux.sh
#    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

feat = net.blobs['fc7'].data[4]
print feat.shape[0]

np.savetxt('feat7.txt', feat.reshape(1, 4096), fmt = '%-8.4f')


plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.savefig('hist.png')

# sort top k predictions from softmax output
top_k = net.blobs['prob'].data[4].flatten().argsort()[-1:-6:-1]
print labels[top_k]