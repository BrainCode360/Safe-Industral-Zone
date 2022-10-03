import cv2
import numpy as np

  
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



'''
"nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"'''

def main():
    # parse arguments
    cv2.namedWindow("Display Cams", cv2.WINDOW_NORMAL)

    # setup processor and visualizer
    #processor = Processor(model=args['model'])

    # open video streams sequentially
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

    #  fetch input
    i = 1
    while(True):

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
        
        if i == 40:
            cv2.imwrite('left.jpg', frame)
            cv2.imwrite('back.jpg', frame1)
            cv2.imwrite('right.jpg', frame2)
            cv2.imwrite('front.jpg', frame3)

        
        upper = np.hstack((frame, frame1))
        lower = np.hstack((frame2, frame3))
       
        merged_frame = np.vstack((upper, lower))
        # final_frame= imutils.resize(merged_frame, width=1600)
        final_frame = cv2.resize(merged_frame, (1600, 1000), interpolation=cv2.INTER_LINEAR)

        cv2.imshow('Results', final_frame)
        i += 1
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()   
