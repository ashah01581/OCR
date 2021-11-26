
import cv2
import numpy as np
import os

file = 'rough2.jp2'
final_file = 'final_'+file
image = cv2.imread(file)
# image = cv2.imread('cv2/1.png')
result = image.copy()

#-------------- to get image size
from PIL import Image
width, height = Image.open('rough1.jp2').size
# print('width:',width,'height:',height)


#----------------------------- detect lines
window_name = 'image'
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
# cv2.imwrite("/Users/aki/Downloads/OCR/Rough/cv2/first_result.png", thresh)

# Detect horizontal lines
# ------ getStructuringElement (MorphShapes shape,Size (x axis, yaxis))
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
# cv2.imwrite("/Users/aki/Downloads/OCR/Rough/cv2/first_result.png", detect_horizontal)
cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(result, [c], -1, (36,255,12), 1)


# # Detect vertical lines
# -------- getStructuringElement (MorphShapes shape,Size (x axis, yaxis))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
# cv2.imwrite("/Users/aki/Downloads/OCR/Rough/cv2/first_result.png", detect_vertical)
cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    # with open("/Users/aki/Downloads/OCR/Rough/cv2/_cordinates", 'a') as file1:
    #     file1.write(str(c))
    cv2.drawContours(result, [c], -1, (36,255,12), 1)

cv2.imwrite("/Users/aki/Downloads/OCR/Rough/cv2/"+final_file, result)

#------ after detecting lines get co-ordinates in asc order and detects group of lines
out=[]
for c in cnts:
    for x in range(len(c)):
        out.append(c[x][0].tolist())
out.sort(key=lambda k: (k[0], k[1]))

z=[]
for x in range(len(out)-1):
    y=(out[x+1][0] - out[x][0])
    if y > 24:                      #if diff > 24 its different line
        z.append(out[x])
        z.append('*')
    else:
        z.append(out[x])

with open("/Users/aki/Downloads/OCR/Rough/_cordinates", 'w') as file1:
    file1.write(str(z))
# cv2.imwrite("/Users/aki/Downloads/OCR/Rough/"+final_file, result)

#---------------------------------------- to draw line
# import cv2

# path = r"/Users/aki/Downloads/OCR/Rough/cv2/"+final_file
# # Reading an image in default mode
# image = cv2.imread(path)

# # Window name in which image is displayed
# window_name = 'Image'

# # Start coordinate, here (0, 0) represents the top left corner of image
# start_point = (50, 50)

# # End coordinate, here (250, 250) represents the bottom right corner of image
# end_point = (150, 50)

# # pink color in BGR
# color = (255,0,255)
# thickness = 1

# # Using cv2.line() method Draw a diagonal pink line with thickness of 9 px
# image = cv2.line(image, start_point, end_point, color, thickness)

# # Displaying the image
# cv2.imwrite("/Users/aki/Downloads/OCR/Rough/cv2/final_rough3_2.jp2", image)

#--------------- to get coordination by color
# import cv2  
# import numpy as np
# # Load image
# im = cv2.imread("/Users/aki/Downloads/OCR/Rough/cv2/final_rough3_2.jp2")

# # Define the blue colour we want to find - remember OpenCV uses BGR ordering
# green = [255,0,255]

# # Get X and Y coordinates of all blue pixels
# X,Y = np.where(np.all(im==green,axis=2))
# zipped = np.column_stack((Y,X))
# for x in zipped:
#     with open("/Users/aki/Downloads/OCR/Rough/cv2/_cordinates", 'a') as file1:
#         file1.write(str(x))


#------------------------------

