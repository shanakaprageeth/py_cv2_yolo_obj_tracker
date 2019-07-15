__author__ = 'shanaka prageeth'
__description__ = 'This script identify labels in video streams'

# common packages
import datetime
import argparse
import logging
# ML packages
import numpy as np
import math
# image processing package
import cv2
import imutils

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', default='test2.mp4',
                    help='video stream')
parser.add_argument('-c', '--config', default='yolov3.cfg',
                    help='yolo config file')
parser.add_argument('-w', '--weights', default='yolov3.weights',
                    help='yolo weights file')
parser.add_argument('-l', '--labels', default='coco.names',
                    help='line seperated label names file')
parser.add_argument('-sx', '--sizex', default=416, type=int,
                    help='DNN input size')
parser.add_argument('-sy', '--sizey', default=416, type=int,
                    help='DNN input size')
args = parser.parse_args()

# logger setup
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(level=logging.INFO,format=FORMAT)


class Track_Object:
    last_loc_x=-1
    last_loc_y=-1
    obj_type=-1

    def __init__(self, obj_type, last_loc_x,last_loc_y):
        self.obj_type = obj_type
        self.last_loc_x = last_loc_x
        self.last_loc_y = last_loc_y

def main():
    # is cv2 optimized
    if(not cv2.useOptimized()):
        cv.setUseOptimized(True)
        logging.info("set optimized {0}".format(cv2.useOptimized()))

    # CNN spatial input image scaling coeifficents
    in_width = args.sizex #YOLO input
    in_height = args.sizey
    # 1/alpha
    scale_factor = 1 / 255.0
    # RGB mean substraction
    RGB_mean = (0,0,0)
    # swap RB channels
    swapRB = True

    # on object numbers
    total_in_screen = 0
    text_in = 0
    text_out = 0
    # track objects

    # capture the video stream
    cap = cv2.VideoCapture(args.video)
    # load labels names
    labels = []
    with open(args.labels, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    # create random color space for labels
    COLORS = np.random.uniform(0, 255, size=(len(labels), 3))
    # customized label list for filtering
    filter_labels = ['person'] #labels
    # to improve timing set flags
    filter_label_active = [True if (itm in filter_labels) else False for itm in labels]
    for idx,itm in enumerate(labels):        
        logging.debug(itm, filter_label_active[idx])
    # read DNN into cv2 dnn format
    net = cv2.dnn.readNet(args.weights, args.config)
    # identify output layer of the NN
    output_layer = net.getUnconnectedOutLayersNames()

    while (True):        
        start_t = cv2.getTickCount()
        total_in_screen = 0
        # Capture frame-by-frame
        ret, frame = cap.read()
        # use follwoing line if Umat
        ret, Oframe = cap.read()
        if not ret:
            logging.error("failed to capture the frame")
            break
        # umat should make things faster but difficult to workwith in python 
        # should use C++ cuda
        #frame = cv2.UMat(Oframe)
        # resize the frame, convert it to grayscale, and blur it
        frame = cv2.resize(frame, (in_height,in_width))
        [frame_height, frame_width, *frame_rest] = frame.shape
        # use follwoing line if Umat
        #[frame_height, frame_width, *frame_rest] = Oframe.shape
        # preprocessing the image to size for CNN spatial size
        blob = cv2.dnn.blobFromImage(
            frame, scale_factor, (in_width, in_height), RGB_mean, swapRB, crop=False)
        # set DNN inputs with scaled values
        net.setInput(blob)
        # DNN forward propogation. Can use new Aync and optional [,output] for few OP detection
        dnn_outputs = net.forward(output_layer)

        # output filtration        
        conf_threshold = 0.5
        nms_threshold = 0.4
        label_idxs = []
        boxes = []
        confidences = []
        # loop over each layer outputs
        for dnn_output in dnn_outputs:
            # loop over each detection of blobs
            for detection in dnn_output:
                # select the best label with highest output probability for blob
                # first 4 indexes are blob dimensions
                detection_val = detection[5:]
                label_idx = np.argmax(detection_val)
                if(not filter_label_active[label_idx]):
                    pass
                else:
                    confidence = detection_val[label_idx]
                    if confidence > conf_threshold:
                        # add to total persons
                        total_in_screen+=1
                        # scale the bounding box to frame
                        (cx,cy,w,h) = map(int,detection[0:4]*np.array([frame_width,frame_height, frame_width,frame_height]))
                        x = int(cx - w / 2)
                        y = int(cy - h / 2)
                        
                        # add predictions to lists for NMS normalization
                        label_idxs.append(label_idx)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])                        

        # non-maxima suppression to suppress overlapping blobs/boxes
        detects_normalized = cv2.dnn.NMSBoxes(
            boxes, confidences, conf_threshold, nms_threshold)
        # loop over detections left after normalization
        for i in detects_normalized:
            i = i[0]
            [x,y,w,h,*rest] = boxes[i]
            label = str(labels[label_idxs[i]])
            color = COLORS[label_idxs[i]]
            # mark center point of the objext
            cv2.circle(frame, (int(cx), int(cy)), 1, (0, 0, 255), 5)
            # draw a rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
            # draw label
            cv2.putText(frame, label, (x-10, y-10),cv2.FONT_HERSHEY_TRIPLEX, 0.2, color, 1)

            # tracking

        # fps calculation
        fps = 1/((cv2.getTickCount()-start_t)/cv2.getTickFrequency())        

        # print values on frame
        text_start_x = frame_width - 100
        text_start_y = 30
        text_seperation = 20
        cv2.putText(frame, "Total: {}".format(str(total_in_screen)), 
            (text_start_x, text_start_y+text_seperation*1),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "Inward: {}".format(str(text_in)), 
            (text_start_x, text_start_y+text_seperation*2),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "Outward: {}".format(str(text_out)), 
            (text_start_x, text_start_y+text_seperation*3),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "FPS: {}".format(str(round(fps,2))), 
            (text_start_x, text_start_y+text_seperation*4),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # clean up release and distroy opencv windows 
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()