# imports required libraries
import cv2
import numpy as np
import argparse
import time
import os


def imageDetection(im, y, c, t):
    # Load the COCO class Labels which our YOLO model will be trained on
    labelsPath = y + '\coco.names'
    LABELS = open(labelsPath).read().strip().split('\n')
    print(labelsPath)

    # Create random colours to represent possible Labels
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

    # Get the YOLO weights and config from entered path
    weightsPath = y + '\yolov3.weights'
    configPath = y + '\yolov3.cfg'

    # Loud the YOLO trained on the COCO dataset into a DNN(Deep, Neural Network )
    print('[INFO] Loading YOLO data')
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Load input image and grab the dimensions
    image = cv2.imread(im)
    (H, W) = image.shape[:2]

    # Determine the output layer names from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # Construct a blob from the input image and then perform a forward pass of YOLO
    # Which provides the bounding boxes and probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # Show the time it took for YOLO to complete
    print('[INFO] YOLO took: {:.6f} seconds'.format(end-start))

    # Creates lists for the bounding  boxes, confidence and class ID's
    boxes = []
    confidences = []
    classIDs = []

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
                # draw the bounding box and the label
            colour = [int(c) for c in colours[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
            text = '{}: {:.4f}'.format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    cv2.imshow('test', image)
    cv2.waitKey(0)




