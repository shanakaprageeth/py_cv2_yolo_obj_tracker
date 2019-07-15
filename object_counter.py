__author__ = 'shanaka prageeth'
__description__ = 'This script identify objects in video streams'

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
logging.basicConfig(level=logging.INFO)

# save_csv
SAVE_CSV = True
# track objects
TRACK = True
# save output video
VID_OUT = True

class Track_Object:    
    obj_type=-1
    last_loc_x=-1
    last_loc_y=-1
    dir_x=0
    dir_y=0
    def __init__(self, obj_type, last_loc_x,last_loc_y):
        self.obj_type = obj_type
        self.last_loc_x = last_loc_x
        self.last_loc_y = last_loc_y

def main():
    if(SAVE_CSV):
        fp = open("output.log","w") 
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
    total_in = 0
    total_out = 0
    # track objects
    tracked_objects = []
    tracked_objs_in = []
    tracked_objs_out = []

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
        logging.debug(itm)
        logging.debug(filter_label_active[idx])
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
        # ret, Oframe = cap.read()
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
        vid_init = None
        out = None
        if(VID_OUT and not vid_init):
            # opencv write video to a file.
            vid_out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
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
            cv2.putText(frame, label, (x, y-15),cv2.FONT_HERSHEY_TRIPLEX, 0.4, color, 1)
            if(SAVE_CSV):
                fp.write("{},{},{},{},{},{}\n".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, cx, cy, w, h))
            # tracking
            # if no object is there
            if(TRACK):
                #tracking variables
                max_move = frame_width//8
                # divide the frame by 2 on x direction
                margin_x = frame_width //2
                margin_y = frame_height
                obj_idx = 0
                if (len(tracked_objects)==0):
                    tracked_objects.append(Track_Object(label_idxs[i], cx,cy))
                # if objects are available
                else:
                    match_found = False
                    for idx,track_object in enumerate(tracked_objects):
                        # check objects that were close for a match
                        dir_x = track_object.last_loc_x - cx
                        if (abs(dir_x)<max_move):
                            dir_y = track_object.last_loc_y - cy
                            if (abs(dir_y)<max_move):
                                track_object.dir_x=round(dir_x,2)
                                track_object.dir_y=round(dir_y,2)
                                track_object.last_loc_x=cx
                                track_object.last_loc_y=cy
                                obj_idx = idx
                                match_found = True
                    if(not match_found):
                        obj_idx = len(tracked_objects)
                        tracked_objects.append(Track_Object(label_idxs[i], cx,cy))
                # print additional details about tracking and inward outward counts
                #total_in = 0
                #total_out = 0
                for idx,track_object in enumerate(tracked_objects):
                    cv2.putText(frame, str(idx), (cx+15, cy-+15),cv2.FONT_HERSHEY_TRIPLEX, 0.4, color, 1)
                    track_details = "x` :{1}   y`:{2}".format(idx,track_object.dir_x, track_object.dir_y)
                    cv2.putText(frame, track_details, (x, y-5),cv2.FONT_HERSHEY_TRIPLEX, 0.4, color, 1)
                    logging.debug(str(idx))
                    logging.debug(track_details)
                    track_details = "x  :{1}   y :{2}".format(idx,track_object.last_loc_x, track_object.last_loc_y)
                    logging.debug(track_details)
                    if(track_object.last_loc_x < margin_x and track_object.last_loc_x+track_object.dir_x > margin_x ):
                        # prevent double entry object already in cannot go again in
                        if(idx not in tracked_objs_in):
                            total_in += 1
                            tracked_objs_in.append(idx)
                    elif(track_object.last_loc_x > margin_x and track_object.last_loc_x+track_object.dir_x < margin_x ):
                        # prevent double entry
                        if(idx not in tracked_objs_out):
                            total_out += 1
                            tracked_objs_out.append(idx)
                    else:
                        # all the other objects that doesn't pass the boundry or stationary
                        pass

                # draw a in out margin
                cv2.line(frame, (margin_x,0),(margin_x, margin_y), (0,0,255), 1)
                    

        # fps calculation
        fps = 1/((cv2.getTickCount()-start_t)/cv2.getTickFrequency())        

        # print values on frame
        text_start_x = frame_width - 100
        text_start_y = 30
        text_seperation = 20
        cv2.putText(frame, "Total: {}".format(str(total_in_screen)), 
            (text_start_x, text_start_y+text_seperation*1),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 2)
        if(TRACK):
            cv2.putText(frame, "Inward: {}".format(str(total_in)), 
                (text_start_x, text_start_y+text_seperation*2),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Outward: {}".format(str(total_out)), 
                (text_start_x, text_start_y+text_seperation*3),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "FPS: {}".format(str(round(fps,2))), 
            (text_start_x, text_start_y+text_seperation*4),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Live Feed", frame)
        if(VID_OUT):
            vid_out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # clean up release and distroy opencv windows
    fp.close()
    cap.release()
    vid_out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()