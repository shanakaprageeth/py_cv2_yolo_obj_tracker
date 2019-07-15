wget https://pjreddie.com/media/files/yolov3.weights
cd ../
git clone https://github.com/pjreddie/darknet
cd darknet
make
cd ../headcounter
cp ../darknet/data/coco.names .
cp ../darknet/cfg/yolov3.cfg .
