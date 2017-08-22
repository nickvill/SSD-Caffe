import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/nick/caffe/'
# sys.path.insert(0, caffe_root + 'python')  
import caffe  
from caffe.model_libs import *
from google.protobuf import text_format
import time
import math
import shutil 
import stat
import subprocess 

caffe.set_device(0)
caffe.set_mode_gpu()

cap = cv2.VideoCapture(0)

net_file=   '/home/nick/caffe/models/MobileNet/VOC0712/SSD_300x300/MobileNet_VOC0712_SSD_300x300.prototxt'  
caffe_model='/home/nick/caffe/models/MobileNet/VOC0712/SSD_300x300/MobileNet_VOC0712_SSD_300x300.caffemodel'  
test_dir = "images"
os.stat(caffe_model)

if not os.path.exists(caffe_model):
# if not os.path.exists('/home/nick/models/MobileNet/VOC0712/MobileNetSSD_deploy.caffemodel'):
	print("MobileNetSSD_deploy.caffemodel does not exist,")
	print("use merge_bn.py to generate it.")
	exit()

# solver_mode = P.Solver.GPU
# gpus = '0'
net = caffe.Net(net_file,caffe_model,caffe.TEST)  



CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    # origimg = cv2.imread(imgfile)
    img = preprocess(imgfile)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(imgfile, out)

    for i in range(len(box)):
    	p1 = (box[i][0], box[i][1])
    	p2 = (box[i][2], box[i][3])
    	cv2.rectangle(imgfile, p1, p2, (0,255,0))
    	p3 = (max(p1[0], 15), max(p1[1], 15))
    	title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
    	cv2.putText(imgfile, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("SSD", imgfile)

frames2count = 60
count = 0 
desired_fps = 20

while(True):

	if count == 0:
		start = time.time()

	if cap.grab():
		
		ret, frame = cap.retrieve()

		start_comp = time.time()
		detect(frame)
		end_comp = time.time()

		sleep_time = (1./desired_fps) - (end_comp - start_comp)
		time.sleep(abs(sleep_time))

		count += 1
	
	if count == frames2count:
		end = time.time()
		seconds = end - start
		fps = frames2count/seconds
		print "fps: {}".format(fps)
		count = 0 

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
