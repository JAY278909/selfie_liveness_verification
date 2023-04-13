import sys
import os 
from Liveness_Verification import *
from PIL import Image
import io


getClass = Liveness_Verification()
im = Image.open("download.jpeg")
img_byte_arr = io.BytesIO()
im.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()

print(getClass.verifyLiveness(img_byte_arr,{}))
