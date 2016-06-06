import cv2 as cv
import os

def save(filename, image, output_path=None):
  dir_name = os.path.dirname(filename)
  
  if output_path and dir_name != output_path:
    filename = os.path.join(output_path, os.path.basename(filename))

  #add extension
  filename = "".join([filename, ".png"])

  cv.imwrite(filename, image)
    
  return filename

def show(f, image):
  cv.imshow(f, image)