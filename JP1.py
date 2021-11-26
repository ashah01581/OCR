import pytesseract
import numpy as np
from PIL import Image

# To save JP2 file in text file.
filename = 'rough13.jp2'
img1 = np.array(Image.open(filename))
text = pytesseract.image_to_string(img1)

#save variable in file
with open('/Users/aki/Downloads/OCR/JP2000/rough13.txt','w') as f:
    f.write(str(text))

