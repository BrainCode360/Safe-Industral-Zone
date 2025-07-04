import numpy as np
import cv2
import time


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def nms(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1e-9) * (y2 - y1 + 1e-9)
    order = scores.argsort()[::-1] # get boxes with more ious first
    
    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        other_box_ids = order[1:]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[other_box_ids])
        yy1 = np.maximum(y1[i], y1[other_box_ids])
        xx2 = np.minimum(x2[i], x2[other_box_ids])
        yy2 = np.minimum(y2[i], y2[other_box_ids])
        
        #print(list(zip(xx1, yy1, xx2, yy2)))

        w = np.maximum(0.0, xx2 - xx1 + 1e-9) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1e-9) # maxiumum height
        inter = w * h
        
        ovr = inter / (areas[i] + areas[other_box_ids] - inter)
        
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def non_max_suppression(prediction, conf_thres=0.45, iou_thres=0.5, classes=None, agnostic=False, multi_label=False, labels=(), max_det=150):


    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
       

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(float)), axis=1)

        else:  # best class only
            conf = np.amax(x[:, 5:], axis=1, keepdims=True)
            j = np.argmax(x[:, 5:], axis=1).reshape(conf.shape)
            x = np.concatenate((box, conf, j.astype(float)), axis=1)[conf.flatten() > conf_thres]
            
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = nms(boxes, scores, iou_thres)  # NMS

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    # print(output)
    return output

def get_scaled_coords(input_size, xyxy, output_image, pad):

        """
        Converts raw prediction bounding box to orginal
        image coordinates.
        
        Args:
          xyxy: array of boxes
          output_image: np array
          pad: padding due to image resizing (pad_w, pad_h)
        """
        pad_w, pad_h = pad
        pad_w, pad_h = 0, 0
        in_h, in_w = input_size
        out_h, out_w, _ = output_image.shape
                
        ratio_w = out_w/(in_w - pad_w)
        ratio_h = out_h/(in_h - pad_h) 
        
        out = []
        for coord in xyxy:

            x1, y1, x2, y2 = coord
                        
            x1 *= ratio_w # in_w # /ratio_w
            x2 *= ratio_w # in_w # /ratio_w
            y1 *= ratio_h # in_h # /ratio_h
            y2 *= ratio_h # in_h # /ratio_h
            
            x1 = max(0, x1)
            x2 = min(out_w, x2)
            
            y1 = max(0, y1)
            y2 = min(out_h, y2)
            
            out.append((x1, y1, x2, y2))
        
        return np.array(out).astype(int)


def process_predictions(input_size, det, output_image, pad):
        """
        Process predictions and optionally output an image with annotations
        """
        draw_boxes = True
        save_txt = False
        hide_labels = True
        if len(det):
            # Rescale boxes from img_size to im0 size
            # x1, y1, x2, y2=
            det[:, :4] = get_scaled_coords(input_size, det[:,:4], output_image, pad)
            output = {}
            # base, ext = os.path.splitext(output_path)
            
            s = ""
            names = ['person', 'Forklift', 'Truck']
            # Print results
            for c in np.unique(det[:, -1]):
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
            if s != "":
                s = s.strip()
                s = s[:-1]
            
            print("Detected: {}".format(s))
   
            
        return det
