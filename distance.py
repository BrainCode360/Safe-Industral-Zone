from audioop import minmax
from glob import glob1
from logging import warning
import math
import random
import os
import cv2
import numpy as np
import time
import sys
import imutils
from time import sleep
from threading import Thread
from scipy.spatial import distance as dist
import time

import Jetson.GPIO as GPIO


# GPIO.cleanup()



# from inference_image import image_capture

LABELS = ['Person', 'Forklift', 'Truck']
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

global mycnt
mycnt = 0

back_red_zone_points = [(1, 599), (16, 515), (46, 428), (98, 348), 
                (151, 295), (202, 261), (259, 235), 
                (359, 210), (452, 211), (512, 226), 
                (585, 255), (669, 316), (727, 382),
                (768, 463), (793, 598)]

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length


# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


# Distance constants 
KNOWN_DISTANCE =  5.0 # METERS
PERSON_WIDTH = 1.0 # METER
FORKLIFT_WIDTH = 4.0 # METERS
TRUCK_WIDTH= 200.0 # METERS

# reading the reference image from dir

# ref_person = cv2.imread('person.png')
# ref_forklift = cv2.imread('forklift.png')
# ref_person = cv2.imread('truck.png')


# person_data = image_capture(ref_person)
# person_width_in_rf = person_data[0][1]

# forklift_data = image_capture(ref_forklift)
# forklift_width_in_rf = forklift_data[0][1]

# truck_data = image_capture(ref_person)
# truck_width_in_rf = truck_data[0][1]

# print(f"Person width in pixels : {person_width_in_rf} ForkLift width in pixel: {forklift_width_in_rf} Truck width in pixel: {truck_width_in_rf}")


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def conv_to_xywh(xmin, ymin, xmax, ymax):
    x = xmin
    y = ymin
    w = xmax - xmin
    h = ymax - ymin
    return x, y, w, h


def cvDrawBoxes(detections, img, polygone):

    GPIO.setmode(GPIO.BOARD)
    channel = 13
    GPIO.setup(channel, GPIO.OUT, initial=GPIO.LOW)

    global mycnt
    listx = []
    listy = []
    listacc = []
    warning_text = 'RED ZONE'

    object_centers = []


    idetec = len(detections)
    i = 0
    alpha = 0.4

    data_list = []

    # finding focal lengths

    focal_person = 10 # focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
    focal_forklift = 20 # focal_length_finder(KNOWN_DISTANCE, FORKLIFT_WIDTH, forklift_width_in_rf)
    focal_truck = 30 # focal_length_finder(KNOWN_DISTANCE, TRUCK_WIDTH, truck_width_in_rf)

    for *xyxy, conf, cls in reversed(detections): # loop to populate list of detctions and drawig bounding boxes on detected object
    # for detection in detections: 
        overlay = img.copy() # image copied to draw transparet boundin boxes on detected objects
        box = xyxy
        # box = result_boxes[0]
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        cls = int(cls)
        if cls == 0: # person class id
            cls_text = 'Person'
            distance = distance_finder(focal_person, PERSON_WIDTH, (box[0][1]))
            # data_list.append([LABELS[cls], box[2], (box[0], box[1]-2)])

        elif int(cls) == 1:
            cls_text = 'Forklift'
            distance = distance_finder(focal_forklift, FORKLIFT_WIDTH, (box[0][1]))
            # data_list.append([LABELS[cls], box[2], (box[0], box[1]-2)])

        elif int(cls) == 2:
            distance = distance_finder(focal_truck, TRUCK_WIDTH, (box[0][1]))
            cls_text = 'Truck'
            # data_list.append([LABELS[cls], box[2], (box[0], box[1]-2)])
        
        # if cls != 'person': # skipping other classe except person class
        #     continue
        # print(data_list)
        xmin, ymin, xmax, ymax = int(box[0][0]), int(box[0][1]), int(box[0][2]), int(box[0][3])
        four_points = [(xmin,ymax), (xmax, ymax), (xmin, ymin), (xmax,ymin)]
        results = []
        result = None

        for point in four_points:
            result = cv2.pointPolygonTest(polygone, point, True) # Finding if point lies in polygone
            results.append(result)

        x, y, w, h = conv_to_xywh(xmin, ymin, xmax, ymax)
        cx, cy = (x + (w / 2), y + (h / 2))
        # print(round(conf * 100, 2))

        # for yolov3-tiny
        # xmin = xmin-5
        # ymin = ymin-20
        # xmax = xmax+5
        # ymax = ymax+20

        pt1 = (xmin, ymin) # making tuple
        pt2 = (xmax, ymax)
        
        if round(conf * 100, 2) > 5: # check for accuracy of detected objects greater than 30 percent
            object_centers.append((cx, cy))

            # listx.append(x) # appending each detect object centroid x in a list for further computaions
            # listy.append(y) # appending eachdetect object centroid y in a list for further computaions
            listacc.append(round(conf * 100, 2)) # appending acuuracy rate of each detected object for further computations
            # mimg1 = img.copy() # image copied for point visvualization of detecd objects
            # JUst for shashka purpose
            linecol = (0, 0, 255)
            thk  = 2
            cv2.line(img,(int((xmin)), int(ymin)),(int((xmin+5)), int(ymin)),(linecol),thk) # outlines of bounding boxes
            cv2.line(img,(int((xmin)), int(ymin)),(int((xmin)), int(ymin+5)),(linecol),thk)
            cv2.line(img,(int((xmax)), int(ymin)),(int((xmax-5)), int(ymin)),(linecol),thk)
            cv2.line(img,(int((xmax)), int(ymin)),(int((xmax)), int(ymin+5)),(linecol),thk)

            cv2.line(img,(int((xmin)), int(ymax)),(int((xmin)), int(ymax-5)),(linecol),thk)
            cv2.line(img,(int((xmin)), int(ymax)),(int((xmin+5)), int(ymax)),(linecol),thk)
            cv2.line(img,(int((xmax)), int(ymax)),(int((xmax-5)), int(ymax)),(linecol),thk)
            cv2.line(img,(int((xmax)), int(ymax)),(int((xmax)), int(ymax-5)),(linecol),thk)
            cv2.circle(img, (int(cx), int(cy)), 3, (255,255,0), -1) # centroid drawn for each detected object
            cv2.rectangle(overlay, pt1, pt2, (0,100,0), -1) # transparent bounding box
            
            for result in results:
                if result >= 0:
                    GPIO.output(channel, GPIO.HIGH)
                    warning_text = '' + cls_text + ' CROSSED STOP!'
                    cv2.putText(img, warning_text, (160,480), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 300))
                    cv2.rectangle(overlay, pt1, pt2, (0,0,255), -1) # transparent bounding box
                    break

                # else:
            #     cv2.rectangle(overlay, pt1, pt2, (0,0,255), -1) # transparent bounding box

            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 1, img) # alpha value to set tranparency of bounding boxes low or high(for main frame alpha =0,3)
                # 1, img)
            # cv2.addWeighted(overlay, 0, mimg1, 0, 1, mimg1) # alpha set to zero for point visualizatio frame
        i = i+1
    distance_list = []
    minD = 50
    mPoint = None
    mCenter = None
    minX = None
    minY = None
    # loop to find distance between each detected object red-zone boundery points

    for center in object_centers:
        for point in polygone:
            D = dist.euclidean((center[0], center[1]), (point[0], point[1])) / 0.955 #returns Euclidean distance between vectors u and v (passed centroids of two object to calculate distance)
            D = D / 39.370 # converting distance into meters.
            (mX, mY) = midpoint((center[0], center[1]), (point[0], point[1]))#midpoint to display center between two objects


            # Objects with distance less than 1 meter
            if D <= minD:
                minD = D
                mCenter = center
                mPoint = point
                minX = mX
                minY = mY
            # print(minD)
            # distance_list.append(D)

        if minD <= 20 and D > 15:
            warning_text = '\n Warning:' + cls_text + ' is approching'
            cv2.putText(img, "{:.1f}m".format(D), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0, 0, 255], 1)
            cv2.circle(img, (int(center[0]), int(center[1])), 3, [0, 0, 255], -1)
            cv2.circle(img, (int(point[0]), int(point[1])), 3,  [0, 0, 255], -1)
            cv2.line(img, (int(center[0]), int(center[1])), (int(point[0]), int(point[1])),[0, 0, 255,], 1)
            # cv2.putText(img, 'RED ZONE', (250, 400), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255))

        if minD <= 15: # check if distance is less than 10 meter.

            cv2.putText(img, "{:.1f}m".format(D), (int(minX), int(minY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [255, 0, 0], 1)
            cv2.circle(img, (int(mCenter[0]), int(mCenter[1])), 3, [255, 0, 0], -1)
            cv2.circle(img, (int(mPoint[0]), int(mPoint[1])), 3, [255, 0, 0], -1)
            cv2.line(img, (int(mCenter[0]), int(mCenter[1])), (int(mPoint[0]), int(mPoint[1])),[255, 0, 0], 1)
            # cv2.putText(mimg1, "{:.1f}m".format(D), (int(mX), int(mY - 10)),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.3, [255, 0, 0], 1)
            # cv2.circle(mimg1, (int(center[0]), int(center[1])), 3, [255, 0, 0], -1)
            # cv2.circle(mimg1, (int(point[0]), int(point[1])), 3, [255, 0, 0], -1)
            # cv2.circle(mimg1, (int(center[0]), int(center[1])), 20, [255, 0, 0], 1)
            # cv2.line(mimg1, (int(center[0]), int(center[1])), (int(point[0]), int(point[1])),
            #     [255, 0, 0], 1)
        #Objects with distance less than 2 meters(neighbour objects)
                
    # GPIO.setmode(GPIO.BOARD)
    GPIO.output(channel, GPIO.LOW)
    GPIO.cleanup()
    warning_text = ''
    cv2.putText(img, 'RED ZONE', (250, 400), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255))
    try:
        return img # ,mimg1
    except:
        mimg1 = img
        return img # ,mimg1
