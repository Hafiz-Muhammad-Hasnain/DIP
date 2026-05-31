import cv2
import numpy as np

class AIEngine:
    _model = None

    @staticmethod
    def smart_binarize(img_cv):
        try:
            import rembg
            # rembg.remove handles numpy arrays
            output = rembg.remove(img_cv)
            if len(output.shape) == 3 and output.shape[2] == 4:
                # Alpha channel is the mask
                mask = output[:, :, 3]
            else:
                gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            return mask
        except ImportError:
            return "rembg Not Installed"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def classify_object(crop_img):
        try:
            import tensorflow as tf
            if AIEngine._model is None:
                AIEngine._model = tf.keras.applications.MobileNetV2(weights='imagenet')
            
            # Preprocess the cropped image
            resized = cv2.resize(crop_img, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            x = np.expand_dims(rgb, axis=0)
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
            
            # Predict
            preds = AIEngine._model.predict(x, verbose=0)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]
            
            return str(decoded[0][1])
        except ImportError:
            return "Not Installed"
        except Exception as e:
            return "Error"
