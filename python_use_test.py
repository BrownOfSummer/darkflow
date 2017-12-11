from darkflow.net.build import TFNet
import cv2
import time
import glob

#options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
options = {"model": "mycfg/tiny-yolo-cel.cfg", "load": -1, "threshold": 0.5}
#options = {"model": "mycfg/yolo-cel.cfg", "load": 18600, "threshold": 0.5}
options = {"pbLoad": "tiny-model/tiny-8250-pb/tiny-yolo-cel.pb", "metaLoad": "tiny-model/tiny-8250-pb/tiny-yolo-cel.meta", "summary": None, "threshold": 0.5}
#options = {"pbLoad": "yolo2-model/yolo-cel-18600-pb/yolo-cel.pb", "metaLoad": "yolo2-model/yolo-cel-18600-pb/yolo-cel.meta", "summary": None, "threshold": 0.5}

load_start = time.time()
tfnet = TFNet(options)
load_end = time.time()
img_list=glob.glob("Cel-img/*.jpg")
for imageName in img_list:
    #imgcv = cv2.imread("./Cel-img/output1_0457.jpg")
    imgcv = cv2.imread(imageName)
    result = tfnet.return_predict(imgcv)
    print(result)

test_end = time.time()
print "Load model cost: {}s".format(load_end - load_start)
print "Test cost: {}s / {} imgs".format(test_end - load_end, len(img_list))
