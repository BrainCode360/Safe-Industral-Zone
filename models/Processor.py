import cv2 
import sys
import os
import ctypes
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import multiprocessing as mp
from multiprocessing import Process
from threading import Thread, enumerate
from queue import Queue
from drawing_arc_boundery import draw_redZone_Back
from distance import cvDrawBoxes
import pickle
# from Pridict import YoLov5Process
# import cupy as cp
import math
import time

ctypes.CDLL('/home/rpt/Udentify/DeepStream5/tensorrtx/yolov5/build/libmyplugins.so')
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4

class Processor():
    def __init__(self, model):

        # post processing config
        # process = YoLov5Process()
        anchors = np.array([
            [[10,13], [16,30], [33,23]],
            [[30,61], [62,45], [59,119]],
            [[116,90], [156,198], [373,326]],
        ])
        self.nl = len(anchors)
        self.nc = 3 # classes
        self.no = self.nc + 5 # outputs per anchor
        self.na = len(anchors[0])
        a = anchors.copy().astype(np.float32)
        a = a.reshape(self.nl, -1, 2)
        self.anchors = a.copy()
        self.anchor_grid = a.copy().reshape(self.nl, 1, -1, 1, 1, 2)
        filters = (self.nc + 5) * 3
        self.output_shapes = [(1, 6375, 8)]    # [(1, 3, 80, 80, 85), (1, 3, 40, 40, 85), (1, 3, 20, 20, 85)] [(1, 6375, 8)]
        self.strides = np.array([8., 16., 32.])
        self.img = None
        self.output = None
        self.frame_Type = None
        self.frame_queue = Queue(maxsize=1000)
        self.output_queue_L = mp.Queue(maxsize=1)
        self.output_queue_R = mp.Queue(maxsize=1)
        self.output_queue_F = mp.Queue(maxsize=1)
        self.output_queue_B = mp.Queue(maxsize=1)
        self.det = None
        
        # with open('back_polygone_pts.txt', 'rb') as f:
          #  self.back_polygone_pts = np.array(pickle.load(f), np.int32)
        
        #with open('left_polygone_pts.txt', 'rb') as f:
         #   left_polygone_pts = pickle.load(f)
        
        # with open('right_polygone_pts.txt', 'rb') as f:
          #  right_polygone_pts = pickle.load(f)
        
        # with open('front_polygone_pts.txt', 'rb') as f:
          #  self.front_polygone_pts = np.array(pickle.load(f),np.int32)
        
        self.back_polygone_pts = np.array([[1, 599], [16, 515], [46, 428],[98, 348], 
                [151, 295], [202, 261], [259, 235], 
                [359, 210], [452, 211], [512, 226], [585, 255],
                [669, 316], [727, 382],
                [768, 463], [793, 598]], np.int32)

        self.front_polygone_pts = np.array([[168, 598], [176, 150], [251, 136], [301, 135],
                    [362, 133], [427, 142],
                    [484, 158], [566, 193],
                    [624, 233], [691, 299], [747, 383], [776, 451], [780, 470],[796, 533], [799, 598]], np.int32)


        print('setting up Yolov5n.trt processor with Batchsize 1')

        # load tensorrt engine

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        TRTbin = '{0}/models/{1}'.format(os.path.dirname(__file__), model)

        with open(TRTbin, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

        # allocate memory
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        # save to class
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream
        self.output = None
        for binding in engine:
            # print('trt volume input ', engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding))
            print('model input size ', size)
            self.input_w = 320 #engine.get_binding_shape(binding)[-1]
            self.input_h = 320 #engine.get_binding_shape(binding)[-2]
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            self.batch_size = engine.max_batch_size
            
            if engine.binding_is_input(binding):
                inputs.append({ 'host': host_mem, 'device': device_mem })
            else:
                outputs.append({ 'host': host_mem, 'device': device_mem })
        self.detect_break = True
        # self.processes = list()
        Thread(target=self.post_process, args=(self.output, None, None)).start()
        # self.post_process2 = mp.Process(target=self.process.post_process, args=(self.output, self.img.shape[0], self.img.shape[1], self.frame_Type, self.output_Frame))
        # self.post_process1.start()
        
        # self.post_process2.start()
        # self.processes.append(self.post_process1)
        # self.processes.append(self.post_process2)
        
        #for process in self.processes:
        # self.post_process1.join()
    
    
    def detect(self, img, frame_Type):

        resized = self.pre_process(img)
        output = self.inference(resized)
        # start = time.time()
        self.output = output
        self.img = img
        self.frame_Type = frame_Type
        
        # Reshape to a two dimentional ndarray np.reshape(output[1:], (-1, 6))[:num, :]
        # det = self.post_process(output[0 * 6001: (0 + 1) * 6001], origin_h=img.shape[0], origin_w=img.shape[1])
        # print('NMS execution time:', 1 / (time.time() - start))
        print('putting frames in queue')
        self.frame_queue.put([self.img, self.frame_Type, self.output])
	
        # while(not(self.output_Frame_queue.empty())):
        #     output_frame = self.output_Frame_queue.get()
        
        # output = np.array(output[0])
        # num = int(output[0])
        # reshape from flat to (1, 3, x, y, 85)
        # reshaped = np.reshape(output[1:], (-1, 6))[:1, :] # .reshape(1, 6375, 8)) # for 1 output tensor
        # reshaped.append(cp.reshape(outputs[0], (1, 6375, 8)))
        # return output_frame

    def pre_process(self, img):
        # print('original image shape', img.shape)
        img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.transpose((2, 0, 1)).astype(np.float16)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        return img


    def inference(self, img):
        img = np.ravel(img)
        self.inputs[0]['host'] = img.astype(np.float32) #np.ravel(img)
        
        # transfer data to the gpu
        # for inp in self.inputs:
        #   cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        # run inference
        start = time.time()

        # self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        self.context.execute_async(batch_size=1,
                          bindings=self.bindings,
                          stream_handle=self.stream.handle)
        end = time.time()
        print('tensor execution time:', end-start)
        
        # fetch outputs from gpu
        #for out in self.outputs:
         #   cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # synchronize stream
        self.stream.synchronize()
        output = self.outputs[0]['host']
        return output
    

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        # print('', x.shape)
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        # print('befre ', y)
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            # print('1', y[:, 0])
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            # print('2', y[:, 2])
            y[:, 1] = x[:, 1] - x[:, 3] / 2 # - (self.input_h - r_w * origin_h) / 2
            # print('3', y[:, 1])
            y[:, 3] = x[:, 1] + x[:, 3] / 2 # - (self.input_h - r_w * origin_h) / 2
            # print('4', y[:, 3])
            # y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 # - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 # - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            # y /= r_h

        return y

    def post_process(self, outputs, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        
        print('is true ', self.detect_break)
        
        if (self.detect_break):
            
            if self.frame_queue.empty():
                print('queue is empty.... passing')
                pass
            
            else:
                print('Got frame doing nms')
                img_data = self.frame_queue.get()
                
                img = img_data[0]
                frame_Type = img_data[1]
                outputs = img_data[2]

                origin_h = img.shape[0]
                origin_w = img.shape[1]

                if (outputs is not None):
                    print('inside NMS process')
                    output = outputs[0 * 6001: (0 + 1) * 6001]
                    num = int(output[0])
                    
                    # Reshape to a two dimentional ndarray
                    pred = np.reshape(output[1:], (-1, 6))[:num, :]
                    
                    # Do nms
                    boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
                    result_boxes = boxes[:, :4] if len(boxes) else np.array([])
                    result_scores = boxes[:, 4] if len(boxes) else np.array([])
                    result_classid = boxes[:, 5] if len(boxes) else np.array([])
                    det = list(zip(result_boxes, result_scores, result_classid))
                    self.output = None

                    if(frame_Type == 'F'):
                        
                        if(not (det == [])):
                            img = draw_redZone_Back(img, self.front_polygone_pts, pt1=(176, 160), pt2=(799, 598))
                            output_Frame = cvDrawBoxes(det, img, self.front_polygone_pts)
                            self.output_queue_F.put(output_Frame)
                        
                        else:
                            self.output_queue_F.put(img)
                    
                    elif(frame_Type == 'B'):
                        
                        if(not (det == [])):
                            img = draw_redZone_Back(img, self.back_polygone_pts, pt1 = (1, 599), pt2 = (793, 598))
                            output_Frame = cvDrawBoxes(det, img, self.back_polygone_pts)
                            self.output_queue_B.put(output_Frame)
                        
                        else:
                            self.output_queue_B.put(img)
                    
                    elif(frame_Type == 'L'):

                        if(not (det == [])):
                            
                            img = draw_redZone_Back(img, self.front_polygone_pts, pt1=(176, 160), pt2=(799, 598))
                            output_Frame = cvDrawBoxes(det, img, self.back_polygone_pts)
                            self.output_queue_L.put(output_Frame)
                        
                        else:
                            self.output_queue_L.put(img)

                    else:
                        
                        if(not (det == [])):
                            img = draw_redZone_Back(img, self.front_polygone_pts, pt1 = (176, 160), pt2=(799, 598))
                            output_Frame = cvDrawBoxes(det, img, self.back_polygone_pts)
                            self.output_queue_R.put(output_Frame)
                        
                        else:
                            self.output_queue_R.put(img)

            # self.output_Frame_queue.put(self.output_Frame)

            # return list(zip(result_boxes, result_scores, result_classid))

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou
    
    def get_scaled_coords(self, origin_h, origin_w, x):

        """
        Converts raw prediction bounding box to orginal
        image coordinates.
        
        Args:
          xyxy: array of boxes
          output_image: np array
          pad: padding due to image resizing (pad_w, pad_h)
        """
        y = np.zeros_like(x)
        # print(x, x.shape)
       
        pad_w, pad_h = 0, 0
        in_h, in_w = self.input_h, self.input_w
        out_h, out_w = origin_h, origin_w
                
        ratio_w = out_w/(in_w - pad_w)
        ratio_h = out_h/(in_h - pad_h) 
        
        out = []
        for coord in x:

            x1, y1, x2, y2, conf, cls = coord
                        
            x1 *= ratio_w # in_w # /ratio_w
            x2 *= ratio_w # in_w # /ratio_w
            y1 *= ratio_h # in_h # /ratio_h
            y2 *= ratio_h # in_h # /ratio_h
            
            x1 = max(0, x1)
            x2 = min(out_w, x2)
            
            y1 = max(0, y1)
            y2 = min(out_h, y2)
            
            out.append((x1, y1, x2, y2, conf, cls))
        # print(out)
        if out == []:
            return y
        else:
            return np.array(out)

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # print(boxes[:, :4].shape)
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        #print('before nms ', boxes)
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        # print('after nms ', keep_boxes)
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        # print('these bbs', boxes, boxes.shape)
        # boxes = self.get_scaled_coords(origin_h, origin_w, boxes)
        return boxes
