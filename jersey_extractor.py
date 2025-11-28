import easyocr
import cv2
import numpy as np

class JerseyExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)

    def extract_number(self, frame, bbox):
        # Crop player
        player_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # Preprocessing for OCR
        gray = cv2.cvtColor(player_img, cv2.COLOR_BGR2GRAY)
        
        # Thresholding to isolate numbers (assuming dark numbers on light jersey or vice versa)
        # We might need to try both normal and inverse thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Read text
        results = self.reader.readtext(thresh)
        
        for (bbox, text, prob) in results:
            if text.isdigit() and prob > 0.5:
                return int(text)
        
        return None
