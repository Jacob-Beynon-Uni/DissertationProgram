# imports required libraries
import threading
import numpy as np
import cv2
import argparse
from time import sleep
from tqdm import tqdm
from threading import Thread
from multiprocessing import Process
import imutils
import time
import os
import sys

def videoDetection(v, o, yo, d, f, c, t):
    # Load the COCO class Labels which our YOLO model will be trained on
    labelsPath = os.path.sep.join([yo, 'coco.names'])
    LABELS = open(labelsPath).read().strip().split('\n')

    # Create random colours to represent possible Labels
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

    # Get the YOLO weights and config from entered path
    weightsPath = yo + '\yolov3.weights'
    configPath = yo + '\yolov3.cfg'

    # Loud the YOLO trained on the COCO dataset into a DNN(Deep Neural Network )
    print('[INFO] Loading YOLO data')
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Determine the output layer names from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # Create the video stream
    vs = cv2.VideoCapture(v)
    writer = None
    (W, H) = (None, None)


    # Try to count total frames in the video
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print('[INFO] {} Total number of frames in video'.format(total))
    except:
        print('[INFO] could not determine # of frames in video')
        print('[INFO] no approx, completion time can be provided')
        total = total - 1

    while True:
        (grabbed, frame) = vs.read()
            # if we cant grab the next frame then we have reached the end of the video
        if not grabbed:
                break
            # Grabs the frame size if we don't have it all ready
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        # Construct a blob from the input image and then perform a forward pass of YOLO
        # Which provides the bounding boxes and probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

            # Creates lists for the bounding  boxes, confidence and class ID's
        boxes = []
        confidences = []
        classIDs = []
        currentFrame = 0
            
            # Loop over each layer of outputs
        for output in layerOutputs:
                # loop over each detection
            for det in output:
                    # extract the required data of the current detection
                scores = det[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                    # filter out the weak predictions
                if confidence > c:
                        # scale the bounding box to the size of the image
                    box = det[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype('int')
                    # use the center coordinates to determine the top and left corner of our box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our lists
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

                    # apply non-maxima suppression to suppress weak bounding boxes
                    idxs = cv2.dnn.NMSBoxes(boxes, confidences, c, t)

                    # ensure there is at least one detection
                    if len(idxs) > 0:
                            # loop over the indexes we are keeping
                        for i in idxs.flatten():
                            (x, y) = (boxes[i][0], boxes[i][1])
                            (w, h) = (boxes[i][2], boxes[i][3])
                            if LABELS[classIDs[i]] == d :
                                # draw the bounding box and the label
                                colour = [int(c) for c in colours[classIDs[i]]]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                                text = '{}: {:.4f}'.format(LABELS[classIDs[i]], confidences[i])
                                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)    
                        # Checks the pointer if its empty
                    if writer is None:
                        # Create our writer pointer
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        writer = cv2.VideoWriter(o, fourcc, f, (frame.shape[1], frame.shape[0]), True)
                        if total > 0:
                            elap = (end - start)
                            ehhhh = elap
                            min = (elap * total) / 60
                            print('[INFO] Single frame took {:.4f} second'.format(elap))
                            print('[INFO] estimated total time to finish {:.4f} Minutes'.format(min))
                    writer.write(frame)
    print('[INFO] cleaning up..')
    writer.release()
    vs.release()


