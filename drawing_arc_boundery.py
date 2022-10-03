from logging import warning
import cv2
import numpy as np


def draw_polygone(pts, image):
    alpha = 0.5 
    pts = pts.reshape((-1, 1, 2))

    # int_coords = lambda x: np.array(x).round().astype(np.int32)
    # exterior = [int_coords(pts.exterior.coords)]
 
    isClosed = True
    
    # Blue color in BGR
    color = (0, 0, 255)
    
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.polylines() method
    # Draw a Red polygon with
    # thickness of 2 px

    image = cv2.polylines(image, [pts], isClosed, color, thickness)

    overlay = image.copy()
    cv2.fillPoly(overlay, pts=[pts], color=(0, 0, 255))

    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    return image_new
    

def convert_arc(pt1, pt2, sagitta):

    # extract point coordinates
    x1, y1 = pt1
    x2, y2 = pt2

    # find normal from midpoint, follow by length sagitta
    n = np.array([y2 - y1, x1 - x2])
    n_dist = np.sqrt(np.sum(n**2))

    if np.isclose(n_dist, 0):
        # catch error here, d(pt1, pt2) ~ 0
        print('Error: The distance between pt1 and pt2 is too small.')

    n = n/n_dist
    x3, y3 = (np.array(pt1) + np.array(pt2))/2 + sagitta * n

    # calculate the circle from three points
    # see https://math.stackexchange.com/a/1460096/246399
    
    A = np.array([
        [x1**2 + y1**2, x1, y1, 1],
        [x2**2 + y2**2, x2, y2, 1],
        [x3**2 + y3**2, x3, y3, 1]])
    M11 = np.linalg.det(A[:, (1, 2, 3)])
    M12 = np.linalg.det(A[:, (0, 2, 3)])
    M13 = np.linalg.det(A[:, (0, 1, 3)])
    M14 = np.linalg.det(A[:, (0, 1, 2)])

    if np.isclose(M11, 0):
        # catch error here, the points are collinear (sagitta ~ 0)
        print('Error: The third point is collinear.')

    cx = 0.5 * M12/M11
    cy = -0.5 * M13/M11
    radius = np.sqrt(cx**2 + cy**2 + M14/M11)

    # calculate angles of pt1 and pt2 from center of circle
    pt1_angle = 180*np.arctan2(y1 - cy, x1 - cx)/np.pi
    pt2_angle = 180*np.arctan2(y2 - cy, x2 - cx)/np.pi

    return (cx, cy), radius, pt1_angle, pt2_angle


def draw_ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness=2, lineType=cv2.LINE_AA, shift=10):
    
    # uses the shift to accurately get sub-pixel resolution for arc
    # taken from https://stackoverflow.com/a/44892317/5087436
    
    center = (
        int(round(center[0] * 2**shift)),
        int(round(center[1] * 2**shift))
    )
    axes = (
        int(round(axes[0] * 2**shift)),
        int(round(axes[1] * 2**shift))
    )
    return cv2.ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness, lineType, shift)


def draw_redZone_Back(img, polygone_pts, pt1, pt2):

    # pt1 = (1, 599)
    # pt2 = (793, 598)


    # polygone_pts = np.array([[1, 599], [16, 515], [46, 428],[98, 348], 
    #                 [151, 295], [202, 261], [259, 235], 
    #                 [359, 210], [452, 211], [512, 226], [585, 255],
    #                 [669, 316], [727, 382],
    #                 [768, 463], [793, 598]],
    #             np.int32)

    # sagitta = 200

    # center, radius, start_angle, end_angle = convert_arc(pt1, pt2, sagitta)
    # axes = (radius, radius)
    # img = draw_ellipse(img, center, axes, 0, start_angle, end_angle, (0, 0, 255))
    image = draw_polygone(polygone_pts , img)
    
    return image





# img = cv2.imread('front.jpg') # np.zeros((500, 500), dtype=np.uint8)
# pt1 = (1, 599)
# pt2 = (793, 598)

# polygone_pts = np.array([[1, 599], [16, 515], [46, 428],[98, 348], 
#                 [151, 295], [202, 261], [259, 235], 
#                 [359, 210], [452, 211], [512, 226], [585, 255],
#                 [669, 316], [727, 382],
#                 [768, 463], [793, 598]],
#                np.int32)


# front_polygone_pts = np.array([[164, 597], [184, 210],[256, 203], [336, 201],
#                 [512, 226], [585, 255],
#                 [669, 316], [727, 382],
#                 [775, 463], [793, 516], [799, 598]],
#                np.int32)

# sagitta = 390

# center, radius, start_angle, end_angle = convert_arc(pt1, pt2, sagitta)
# axes = (radius, radius)
# img = draw_ellipse(img, center, axes, 0, start_angle, end_angle, (0, 0, 255))
# image = draw_polygone(front_polygone_pts , img)

# cv2.imshow('Plotting Red Zone', image)
# cv2.waitKey()


# # center, radius, start_angle, end_angle = convert_arc(pt1, pt2, sagitta)
# # axes = (radius, radius)
# # draw_ellipse(img, center, axes, 0, start_angle, end_angle, 255)
# # center, radius, start_angle, end_angle = convert_arc(pt1, pt2, -sagitta)
# # axes = (radius, radius)
# # draw_ellipse(img, center, axes, 0, start_angle, end_angle, 127)

# cv2.imshow('', img)
# cv2.waitKey()