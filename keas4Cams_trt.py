import cv2
import sys
import argparse
# import ProcessPrediction as pred
from Processor import Processor
# from drawing_arc_boundery import draw_redZone_Back
# from distance import cvDrawBoxes
import numpy as np
import time
import imutils
from threading import Thread, enumerate
from queue import Queue
'''
def inference_frame(processor, frame, frame_Type):
    
    names = ['Person', 'ForkLift', 'Truck']
    
    back_polygone_pts = np.array([[1, 599], [16, 515], [46, 428],[98, 348], 
                [151, 295], [202, 261], [259, 235], 
                [359, 210], [452, 211], [512, 226], [585, 255],
                [669, 316], [727, 382],
                [768, 463], [793, 598]],
               np.int32)


    front_polygone_pts = np.array([[168, 598], [176, 150], [251, 136], [301, 135],
                    [362, 133], [427, 142],
                    [484, 158], [566, 193],
                    [624, 233], [691, 299], [747, 383], [776, 451], [780, 470],[796, 533], [799, 598]],
                np.int32)

    # print(frame.shape)
    output_image = processor.detect(frame, frame_Type) 

    det = pred.process_predictions((320, 320), predict[0], frame, (0, 80))


    if(not(det == [])):

        if(frame_Type == 'F' or 'L' or 'R'):
            img = draw_redZone_Back(frame, front_polygone_pts, pt1=(176, 160), pt2=(799, 598))
            output_image = cvDrawBoxes(det, img, front_polygone_pts)
        
        else:
            img = draw_redZone_Back(frame, back_polygone_pts, pt1 = (1, 599), pt2 = (793, 598))
            output_image = cvDrawBoxes(det, img, back_polygone_pts)
        
    #     
    #     # process detections
    #     for *xyxy, conf, cls in reversed(det):
    #         # Add bbox to image
    #         c = int(cls)  # integer class
    #         label = names[c] # label = self.names[c]
    #         frame = Post_Process.plot_one_box(xyxy, frame, label=label, color=(128, 128, 128), txt_color=(255, 255, 255))
    #  
    #    
    else:
        output_image = frame
    return output_image '''


def cli():
    desc = 'Run TensorRT with yolov5 backbone'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-model',  help='trt engine file located in ./models', required=False)
    parser.add_argument('-cam1', help='video1 path', required=False)
    parser.add_argument('-cam2', help='video2 path', required=False)
    parser.add_argument('-cam3', help='video3 path', required=False)
    parser.add_argument('-cam4', help='video4 path', required=False)
    args = parser.parse_args()
    
    model = args.model or 'yolov5_n6.engine'
    video1 = args.cam1 or 'inputs/0.mp4'
    video2 = args.cam2 or 'inputs/1.mp4'
    video3 = args.cam3 or 'inputs/2.mp4'
    video4 = args.cam4 or 'inputs/3.mp4'

    return { 'model': model, 'cam1': video1, 'cam2': video2, 'cam3': video3, 'cam4': video4 }


def display_frames(processor, cap, cap1, cap2, cap3):

    

    while(True):
        start = time.time()
        print('Displaying frames')
        output1 = processor.output_queue_F.get()
        output2 = processor.output_queue_B.get()
        output3 = processor.output_queue_L.get()
        output4 = processor.output_queue_R.get()
        # print(output1, output2, output3, output4)
        if output1 is None:
            output1 = np.zeros(640, 480)

        if output2 is None:
            output2 = np.zeros(640, 480)

        if output3 is None:
            output3 = np.zeros(640, 480)

        if output4 is None:
            output4 = np.zeros(640, 480)

        upper = np.hstack((output1, output2))
        lower = np.hstack((output3, output4))
        
        #upper = np.hstack((frame, frame1))
        #lower = np.hstack((frame2, frame3))
       
        merged_frame = np.vstack((upper, lower))
        # final_frame= imutils.resize(merged_frame, width=1280)
        
        final_frame = cv2.resize(merged_frame, (1600, 720), interpolation=cv2.INTER_LINEAR)

        cv2.imshow('Display', final_frame)
        print('Display time: ', 1 / (time.time() - start))
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            processor.detect_break = False
            processor.post_process1.terminate()
            sys.exit(0)
            break

    cap.release()
    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()


def main():
    # parse arguments
    args = cli()
    

    model=args['model']
    processor = Processor(model)

    # opening video streams sequentially

    #--- WebCam1
    '''cap = cv2.VideoCapture("v4l2src device=/dev/video0 !"
                           "video/x-raw, format=(string)UYVY, width=(int)1280, height=(int)960 !" 
                            "videoconvert !  appsink", cv2.CAP_GSTREAMER)'''
    cap = cv2.VideoCapture(0)
    
    #--- WebCam2
    '''cap1 = cv2.VideoCapture("v4l2src device=/dev/video1 !"
                            "video/x-raw, format=(string)UYVY, width=(int)1280, height=(int)960 !" 
                            "videoconvert ! appsink", cv2.CAP_GSTREAMER)'''
    
    cap1 = cv2.VideoCapture(1)
    #--- WebCam3
    cap2 = cv2.VideoCapture(2)
    #--- WebCame4
    cap3 = cv2.VideoCapture(3)

    cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
    (x, y, windowWidth, windowHeight) = cv2.getWindowImageRect("Display")

    print("Origin Coordinates(x,y): ", x, y)
    print("Display Width: ", windowWidth)
    print("Display Height: ", windowHeight)

    #cap = cv2.VideoCapture(args['cam1'])
    #cap1 = cv2.VideoCapture(args['cam2'])
    #cap2 = cv2.VideoCapture(args['cam3'])
    #cap3 = cv2.VideoCapture(args['cam4'])

    # display_frames(processor, cap, cap1, cap2, cap3)

    #  fetch input
    
    while(True):

        start = time.time()
        print('taking frames inside')
        frame = cap.read()
        frame1 = cap1.read()
        frame2 = cap2.read()
        frame3 = cap3.read()

        frame = frame[1]
        #print(frame.shape)
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        frame1 = frame1[1]
        frame1 = cv2.resize(frame1, (640, 480), interpolation=cv2.INTER_LINEAR)
        frame2 = frame2[1]
        frame3 = frame3[1]

        # print(frame.shape, frame1.shape, frame2.shape, frame3.shape)
	    
        # inference
        # front = False
        # frame_process1 = mp.Process(target= inference_frame, args=(processor, frame, queue1, front))
        # frame_process2 = mp.Process(target= inference_frame, args=(processor, frame, queue2))
        # frame_process3 = mp.Process(target= inference_frame, args=(processor, frame, queue3))
        # frame_process4 = mp.Process(target= inference_frame, args=(processor, frame, queue4))
        
        output1 = processor.detect(frame, frame_Type='L')
        output2 = processor.detect(frame1, frame_Type='B')
        output3 = processor.detect(frame2, frame_Type='R')
        output4 = processor.detect(frame3, frame_Type='F')

        # start = time.time()
        print('Displaying frames')

        # if (not(processor.output_queue_F.empty())):
        #     output1 = processor.output_queue_F.get()
        # else:
        #     output1 = np.zeros(640, 480)
        
        # if (not(processor.output_queue_B.empty())):
        #     output2 = processor.output_queue_B.get()
        # else:
        #     output2 = np.zeros(640, 480)
        
        # if (not(processor.output_queue_L.empty())):
        #     output3 = processor.output_queue_L.get()
        # else:
        #     output3 = np.zeros(640, 480)
        
        # if (not(processor.output_queue_R.empty())):
        #     output4 = processor.output_queue_R.get()
        # else:
        #     output4 = np.zeros(640, 480)
        
        # print(output1, output2, output3, output4)
        
        if output1 is None:
            output1 = np.zeros(640, 480)

        if output2 is None:
            output2 = np.zeros(640, 480)

        if output3 is None:
            output3 = np.zeros(640, 480)

        if output4 is None:
            output4 = np.zeros(640, 480)

        upper = np.hstack((output1, output2))
        lower = np.hstack((output3, output4))
        
        # upper = np.hstack((frame, frame1))
        # lower = np.hstack((frame2, frame3))
       
        merged_frame = np.vstack((upper, lower))
        # final_frame= imutils.resize(merged_frame, width=1280)
        
        final_frame = cv2.resize(merged_frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

        cv2.imshow('Display', final_frame)
        print('Display time: ', 1 / (time.time() - start))
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            processor.detect_break = False
            # processor.post_process1.terminate()
            sys.exit(0)
            break

    cap.release()
    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()




'''class Post_Process(object):

    def __init__(self) -> None:
        pass

    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.copy()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def plot_one_box(box, im, color=(128, 128, 128), txt_color=(255, 255, 255), label=None, line_width=2):
        # Plots one xyxy box on image im with label
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
        lw = line_width or max(int(min(im.size) / 200), 2)  # line width
        frame_cpy = im.copy()
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(frame_cpy, c1, c2, (0, 0, 255), cv2.FILLED)
        cv2.rectangle(frame_cpy, c1, c2, (0, 0, 255), 3, lineType=cv2.LINE_AA)
        mask = cv2.addWeighted(im, 0.4, frame_cpy, 1-0.5, 0)
    
        if label is not None:
            # print(label)
            tf = max(lw - 1, 1)  # font thickness
            txt_width, txt_height = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            c2 = c1[0] + txt_width, c1[1] - txt_height - 3
            cv2.rectangle(mask, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(mask, label, (c1[0], c1[1] - 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        return mask'''


'''
"nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"'''
