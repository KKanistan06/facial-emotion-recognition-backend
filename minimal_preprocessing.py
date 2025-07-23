import cv2
import numpy as np
import os
from PIL import Image

class MinimalImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.setup_face_detection()
    
    def setup_face_detection(self):
        """Setup face detection with error handling"""
        try:
            # Try multiple paths for cascade file
            cascade_paths = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml', # type: ignore
                'haarcascade_frontalface_default.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
            ]
            
            self.face_cascade = None
            for path in cascade_paths:
                try:
                    if os.path.exists(path):
                        self.face_cascade = cv2.CascadeClassifier(path)
                        if not self.face_cascade.empty():
                            print(f"✅ Face detection initialized with: {path}")
                            break
                except:
                    continue
            
            if self.face_cascade is None or self.face_cascade.empty():
                print("⚠️ Face detection disabled - using center crop")
                self.face_cascade = None
                
        except Exception as e:
            print(f"Face detection setup failed: {e}")
            self.face_cascade = None
    
    def detect_and_crop_face(self, img):
        """Detect face and crop with generous context"""
        if self.face_cascade is None:
            return self.center_crop(img)
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(50, 50)
            )
            
            if len(faces) > 0:
                # Get the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Very generous crop to include full head and shoulders
                cx, cy = x + w//2, y + h//2
                size = int(max(w, h) * 2.2)  # Even more generous
                
                x1 = max(cx - size//2, 0)
                y1 = max(cy - size//2, 0)
                x2 = min(cx + size//2, img.shape[1])
                y2 = min(cy + size//2, img.shape[0])
                
                crop = img[y1:y2, x1:x2]
                print(f"✅ Face detected and cropped: {crop.shape}")
                return crop
            else:
                print("ℹ️ No face detected, using center crop")
                return self.center_crop(img)
                
        except Exception as e:
            print(f"Face detection failed: {e}, using center crop")
            return self.center_crop(img)
    
    def center_crop(self, img):
        """Center crop to square"""
        h, w = img.shape[:2]
        size = min(h, w)
        
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        
        return img[start_y:start_y + size, start_x:start_x + size]
    
    def minimal_preprocessing(self, img):
        """
        Minimal processing to preserve maximum natural quality
        Only does: face detection + crop + resize + grayscale conversion
        """
        try:
            # Step 1: Face detection with generous context
            cropped = self.detect_and_crop_face(img)
            
            # Step 2: Resize using high-quality interpolation
            resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Step 3: Convert to grayscale - NO other processing
            gray_final = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            print(f"✅ Minimal preprocessing complete: {gray_final.shape}")
            return gray_final
            
        except Exception as e:
            print(f"Minimal preprocessing failed: {e}")
            # Fallback: simple resize and grayscale
            try:
                resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                return gray
            except:
                # Ultimate fallback: create dummy image
                return np.full(self.target_size, 128, dtype=np.uint8)
    
    def process_flask_file(self, file_storage):
        """Process image from Flask FileStorage object"""
        try:
            # Read file bytes
            file_storage.seek(0)
            file_bytes = file_storage.read()
            file_storage.seek(0)
            
            # Decode with OpenCV
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
            
            # Apply minimal preprocessing
            processed = self.minimal_preprocessing(img)
            
            # Convert to PIL Image for compatibility with existing transform pipeline
            pil_img = Image.fromarray(processed).convert('RGB')
            
            return pil_img
            
        except Exception as e:
            print(f"Error processing Flask file: {e}")
            # Create fallback PIL image
            dummy = np.full((224, 224, 3), 128, dtype=np.uint8)
            return Image.fromarray(dummy)
    
    def process_image_from_path(self, img_path):
        """Process image from file path"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not load image from {img_path}")
            
            processed = self.minimal_preprocessing(img)
            pil_img = Image.fromarray(processed).convert('RGB')
            return pil_img
            
        except Exception as e:
            print(f"Error processing image from path: {e}")
            dummy = np.full((224, 224, 3), 128, dtype=np.uint8)
            return Image.fromarray(dummy)

# Standalone function for direct use
def minimal_quality_preprocessing(img, target_size=(224, 224)):
    """
    Standalone minimal preprocessing function
    Only does: face detection + crop + resize + grayscale conversion
    """
    processor = MinimalImageProcessor(target_size)
    return processor.minimal_preprocessing(img)
