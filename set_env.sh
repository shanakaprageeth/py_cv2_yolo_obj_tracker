wget https://pjreddie.com/media/files/yolov3.weights
cd ../
git clone https://github.com/pjreddie/darknet
cd darknet
make
cd ../py_cv2_yolo_obj_tracker
cp ../darknet/data/coco.names .
cp ../darknet/cfg/yolov3.cfg .
