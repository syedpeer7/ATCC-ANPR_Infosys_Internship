from ultralytics import YOLO
import cv2
import os
import numpy as np
import base64
import re
from groq import Groq

class ViolationEngine:
    def __init__(self, model_path, plate_model_path, conf=0.25):
        # We assume the Friend's model does EVERYTHING (Bikes, Helmets, Plates)
        print(f"Loading Combined Model: {model_path}")
        self.model = YOLO(model_path)
        
        self.plate_model = self.model 
        self.conf = conf 
        
        # --- GROQ SETUP ---
        # YOUR NEW API KEY (Verified from your request)
        self.groq_client = Groq(api_key="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") 
        
        # --- UPDATED MODEL ID (CRITICAL FIX) ---
        # Old 'llama-3.2-90b' models are deprecated as of April 2025.
        # We must use the new Llama 4 Vision model.
        self.vision_model_id = "meta-llama/llama-4-scout-17b-16e-instruct"
        
        os.makedirs("saved_violations", exist_ok=True)
        self.processed_violation_ids = set()

    def reset(self):
        self.processed_violation_ids.clear()

    def clean_plate_text(self, text):
        # Allow A-Z and 0-9 only
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def encode_image_to_base64(self, image_np):
        _, buffer = cv2.imencode('.jpg', image_np)
        return base64.b64encode(buffer).decode('utf-8')

    def get_plate_text_groq(self, crop_np):
        """Sends image to Groq Llama 4 Vision for reading"""
        base64_image = self.encode_image_to_base64(crop_np)
        try:
            response = self.groq_client.chat.completions.create(
                model=self.vision_model_id, # UPDATED MODEL ID
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Read the license plate text from this image. Output ONLY the alphanumeric characters. Do not add any conversational text."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ],
                    }
                ],
                temperature=0,
            )
            
            raw_text = response.choices[0].message.content
            print(f"Groq Raw Response: {raw_text}") # Debugging print
            return self.clean_plate_text(raw_text)
            
        except Exception as e:
            print(f"Groq API Error: {e}")
            return "Unknown"

    def detect_violations_full_frame(self, frame, frame_count, is_video=True):
        # 1. Run Inference
        results = self.model.track(frame, persist=is_video, conf=self.conf, verbose=False, tracker="botsort.yaml")[0]
        names = results.names
        annotated = frame.copy()
        
        # 2. Categorize Boxes
        bikes = []
        plates = []
        violations = [] 

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().tolist()
            clss = results.boxes.cls.int().cpu().tolist()
            ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else [-1] * len(boxes)
            
            for box, cls, track_id in zip(boxes, clss, ids):
                label = names[cls].lower()
                x1, y1, x2, y2 = map(int, box)
                
                if any(x in label for x in ['motorcycle', 'bike', 'scooter']):
                    bikes.append({'box': [x1,y1,x2,y2], 'id': track_id})
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,255,0), 2)
                    
                elif 'plate' in label:
                    plates.append([x1,y1,x2,y2])
                    
                elif any(x in label for x in ['no helmet', 'no_helmet', 'missing helmet']):
                    violations.append({'box': [x1,y1,x2,y2], 'type': 'No Helmet'})
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 2)
                    
                elif 'mobile' in label or 'phone' in label:
                    violations.append({'box': [x1,y1,x2,y2], 'type': 'Mobile Usage'})
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,0,255), 2)

        current_frame_evidence = []

        # 3. Associate Violations & Plates to Bikes
        for bike in bikes:
            bx1, by1, bx2, by2 = bike['box']
            bike_id = bike['id']
            
            # Check for violations inside this bike box
            active_violation = None
            for v in violations:
                vx1, vy1, vx2, vy2 = v['box']
                vc_x, vc_y = (vx1+vx2)//2, (vy1+vy2)//2
                
                if bx1 < vc_x < bx2 and by1 < vc_y < by2:
                    active_violation = v['type']
                    break 
            
            if active_violation:
                # Violation Confirmed - Find plate
                plate_text = "Unknown"
                best_plate_crop = None
                
                for p_box in plates:
                    px1, py1, px2, py2 = p_box
                    pc_x, pc_y = (px1+px2)//2, (py1+py2)//2
                    
                    if bx1 < pc_x < bx2 and by1 < pc_y < by2:
                        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
                        
                        # Crop Plate with padding
                        h, w = frame.shape[:2]
                        cx1, cy1 = max(0, px1-10), max(0, py1-10)
                        cx2, cy2 = min(w, px2+10), min(h, py2+10)
                        best_plate_crop = frame[cy1:cy2, cx1:cx2]
                        break 
                
                should_process = False
                if not is_video:
                    should_process = True
                elif bike_id != -1 and bike_id not in self.processed_violation_ids:
                    self.processed_violation_ids.add(bike_id)
                    should_process = True
                
                if should_process:
                    if best_plate_crop is not None:
                        # Call Groq here
                        plate_text = self.get_plate_text_groq(best_plate_crop)
                    
                    cv2.putText(annotated, f"{active_violation} [{plate_text}]", (bx1, by1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    evidence_img = frame[by1:by2, bx1:bx2] if best_plate_crop is None else best_plate_crop
                    
                    count_val = len(self.processed_violation_ids) if is_video else np.random.randint(0,1000)
                    cv2.imwrite(f"saved_violations/viol_{frame_count}_{count_val}.jpg", evidence_img)

                    current_frame_evidence.append({
                        'image': evidence_img,
                        'plate': plate_text,
                        'violations': [active_violation]
                    })

        return annotated, current_frame_evidence