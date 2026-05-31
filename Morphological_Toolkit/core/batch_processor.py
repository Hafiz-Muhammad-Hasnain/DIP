import os
import cv2
from .binary_image import BinaryImage
from .morph_op import MorphOp

class BatchProcessor:
    @staticmethod
    def process(input_dir, output_dir, operations, kernel, binarize=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        count = 0
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(input_dir, filename)
                
                bin_img = BinaryImage(file_path)
                if bin_img.image is None: continue
                
                img = bin_img.binarize() if binarize else bin_img.image
                
                for op in operations:
                    if op == 'erode': img = MorphOp.erode(img, kernel)
                    elif op == 'dilate': img = MorphOp.dilate(img, kernel)
                    elif op == 'open': img = MorphOp.opening(img, kernel)
                    elif op == 'close': img = MorphOp.closing(img, kernel)
                    elif op == 'hit-or-miss': img = MorphOp.hit_or_miss(img, kernel)
                    elif op == 'gradient': img = MorphOp.gradient(img, kernel)
                        
                out_path = os.path.join(output_dir, filename)
                cv2.imwrite(out_path, img)
                count += 1
                
        return count
