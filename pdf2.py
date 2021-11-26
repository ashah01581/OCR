# pd.set_option('display.max_rows',100)
# pd.set_option('display.max_columns',100)
# pd.set_option('display.width',227)

import pytesseract
import numpy as np
import pandas as pd

from PIL import Image
from PIL import ImageDraw
import pytesseract
import sys
from pdf2image import convert_from_path
import os
import easyocr
import spacy

PDF_file = "/Users/aki/Downloads/OCR/Rough/rough.pdf"

'''
Part #1 : Converting PDF to images
'''
# Store all the pages of the PDF in a variable
pages= convert_from_path(PDF_file,200)

# Counter to store images of each page of PDF to image
image_counter = 1

# Iterate through all the pages stored above
for page in pages:
    filename = "page_"+str(image_counter)+".jpg"
    page.save(filename, 'JPEG')
    image_counter = image_counter + 1

'''
Part #2 - Recognizing text from the images using OCR
'''
filelimit = image_counter-1

#to save image data as dataframe in file
for i in range(1,filelimit + 1):
    filename = "page_"+str(i)+".jpg"
    img1 = np.array(Image.open(filename))
    text = pytesseract.image_to_data(filename, output_type='data.frame')
    text = text[text.conf != -1]
    text ['text'] = text.groupby(['block_num'])['text'].transform(lambda x: ','.join(x))
    # text['text'] = text.iloc[:,11].drop_duplicates()
    text=text[text.text.notna()]
    # print(text)
    text.to_csv(r'/Users/aki/Downloads/OCR/Rough/rough.txt',index=False,header=True)

#to save image data in textfile
outfile = "/Users/aki/Downloads/OCR/Rough/pdf.txt"
f = open(outfile,"a")
for i in range(1,filelimit + 1):
    filename = "page_"+str(i)+".jpg"
    text = str(pytesseract.image_to_string(Image.open(filename)))
    text = text.replace('-\n','')
    f.write(text)
f.close()
