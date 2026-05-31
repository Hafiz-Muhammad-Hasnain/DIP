import cv2
import numpy as np
from .ai_module import AIEngine

class ShapeAnalyzer:
    @staticmethod
    def analyze(binary_img, orig_img=None, use_ai=False):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
        
        output_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        count = 0
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 10:  # Ignore tiny noise
                cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if use_ai and orig_img is not None:
                    # boundary check
                    crop = orig_img[y:y+h, x:x+w]
                    if crop.size > 0:
                        pred = AIEngine.classify_object(crop)
                        cv2.putText(output_img, pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                count += 1
                
        return output_img, count
