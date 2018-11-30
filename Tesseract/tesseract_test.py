import cv2
import sys
import numpy
import pytesseract
from PIL import Image

if __name__ == '__main__':
 
  #if len(sys.argv) < 2:
  #  print('Usage: python ocr_simple.py image.jpg')
  #  sys.exit(1)
   
  # Read image path from command line
  #imPath = sys.argv[1]
  imPath = "F:\\ML\\Mach-III\Mach-III\\mach3\\Tesseract\\test_label.jpg"
  
  file = Image.open(imPath)
  # Uncomment the line below to provide path to tesseract manually
  # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
 
  # Define config parameters.
  # '-l eng'  for using the English language
  # '--oem 1' for using LSTM OCR Engine
  config = ('-l eng --oem 1 --psm 3')
 
  # Read image from disk
  #im = cv2.imread(file)
 
  # Run tesseract OCR on image
  text = pytesseract.image_to_string(file, lang='eng')
 
  # Print recognized text
  print(text)