import cv2
import numpy as np
import os

# Path to source video:
output_path = 'clicks.txt'



# Mouse callback function
global click_list, img
positions, click_list = [], []
img = None

def draw_polygone(pts, image):
	alpha = 0.5 
	pts = np.array(pts)
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

def callback(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONUP:
		click_list.append([x,y])
		cv2.circle(img, (int(x), int(y)), 9, [255, 0, 55], -1)



images_list = ['back.jpg', 'front.jpg', 'left.jpg' , 'right.jpg']
output_list = ['back.npy', 'front.npy', 'left.npy', 'right.npy']

# Mainloop - show the image and collect the data

for img, output_path in zip(images_list, output_list):
	
	window_name = img.split('.')[0]
	cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
	
	print(window_name)
	
	cv2.namedWindow(window_name)
	cv2.setMouseCallback(window_name, callback)
	img = cv2.imread(img)
	
	while True:
		# img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
		cv2.imshow(window_name, img)

		# Wait, and allow the user to quit with the 'esc' key
		k = cv2.waitKey(1)
		if k == 32:
			if len(click_list) > 0:
				image = draw_polygone(click_list , img)
				img = image
		
		# If user presses 'esc' break 
		if k == 27: break        
	cv2.destroyAllWindows()

	# Write data to a spreadsheet
	# import csv

	with open(output_path, 'wb') as file:
		
		# fieldnames = ['x_position', 'y_position']
		# writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
		# writer.writeheader()
		
		for position in click_list:
			x, y = position[0], position[1]
		np.save(file, np.array(click_list, np.int32))
		# file.write(str(click_list)) # '[' + str(x) + ',' + str(y) + ']')
		click_list = []
	file.close()


def click_track(event, x, y, flags, param):
	global mouse_click, tracker_location,cord,cnt,image,draw_point,lcord
	tracker_location = (x,y)
	# if the left mouse button was clicked, change flag
	# Click is sensed as a button up
	if event == cv2.EVENT_LBUTTONUP:
		mouse_click = True
		lcord = True
