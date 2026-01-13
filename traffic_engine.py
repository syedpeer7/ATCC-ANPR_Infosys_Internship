import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from violation_engine import ViolationEngine

class TrafficLightManager:
    # --- FIX: Accept plate_model_path ---
    def __init__(self, traffic_model_path, violation_model_path, plate_model_path):
        self.model = YOLO(traffic_model_path)
        
        # --- FIX: Pass plate_model_path to ViolationEngine ---
        self.violation_engine = ViolationEngine(violation_model_path, plate_model_path, conf=0.25)
        
        self.track_history = defaultdict(lambda: deque(maxlen=40))
        self.vehicle_directions = {} 
        self.verified_moving = set()
        self.unique_vehicle_classes = {}
        
        self.violation_stats = {"No Helmet": 0, "Mobile Usage": 0, "Triple Riding": 0, "Total Violations": 0}
        self.recent_violations = deque(maxlen=8) 
        
        self.main_lanes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.lane_wait_timers = {d: 0.0 for d in self.main_lanes}
        
        self.weights = {
            'Bus': 2.5, 'Truck': 2.5, 'Car': 1.0, 'SUV': 1.0, 'Sedan': 1.0, 
            'Three-wheeler': 0.8, 'Two-wheeler': 0.5, 'Motorcycle': 0.5, 'Scooter': 0.5
        }
        self.movement_threshold = 5 

    def reset(self):
        self.track_history.clear()
        self.vehicle_directions.clear()
        self.verified_moving.clear()
        self.unique_vehicle_classes.clear()
        self.lane_wait_timers = {d: 0.0 for d in self.main_lanes}
        self.violation_stats = {"No Helmet": 0, "Mobile Usage": 0, "Triple Riding": 0, "Total Violations": 0}
        self.recent_violations.clear()
        self.violation_engine.reset()

    def get_detailed_direction(self, track_id, current_x, current_y):
        if track_id not in self.track_history or len(self.track_history[track_id]) < 10: return None
        prev_x, prev_y = self.track_history[track_id][0]
        dx, dy = current_x - prev_x, current_y - prev_y
        if abs(dx) < self.movement_threshold and abs(dy) < self.movement_threshold: return None 
        return ('DOWN' if dy > 0 else 'UP') if abs(dy) > abs(dx) else ('RIGHT' if dx > 0 else 'LEFT')

    def is_vehicle_stopped(self, track_id):
        history = self.track_history[track_id]
        if len(history) < 20: return False
        start_x, start_y = history[0]
        end_x, end_y = history[-1]
        return np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) < 10 

    def process_frame(self, frame, frame_count, mode="Analysis", is_video=True):
        
        if mode == "Analysis":
            results = self.model.track(frame, persist=True, verbose=False, conf=0.1, tracker="botsort.yaml")
            raw_moving_density = defaultdict(float)
            raw_stopped_density = defaultdict(float)
            vehicle_counts = defaultdict(int)
            annotated_frame = frame.copy()
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.int().cpu().tolist()
                names = results[0].names

                for box, track_id, cls in zip(boxes, track_ids, clss):
                    x, y, w, h = box
                    class_name = names[cls]
                    weight = self.weights.get(class_name, 1.0)
                    
                    rx1, ry1 = int(x - w/2), int(y - h/2)
                    rx2, ry2 = int(x + w/2), int(y + h/2)
                    cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                    
                    self.unique_vehicle_classes[track_id] = class_name
                    vehicle_counts[class_name] += 1
                    self.track_history[track_id].append((float(x), float(y)))
                    
                    start_x, start_y = self.track_history[track_id][0]
                    if np.sqrt((x - start_x)**2 + (y - start_y)**2) > 30:
                        self.verified_moving.add(track_id)
                    
                    raw_direction = self.get_detailed_direction(track_id, x, y)
                    stopped = self.is_vehicle_stopped(track_id)
                    
                    if raw_direction: self.vehicle_directions[track_id] = raw_direction
                    final_direction = raw_direction if raw_direction else (self.vehicle_directions.get(track_id) if track_id in self.verified_moving else None)

                    if final_direction:
                        if stopped: raw_stopped_density[final_direction] += weight
                        else: raw_moving_density[final_direction] += weight
                        cv2.putText(annotated_frame, final_direction, (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            main_lane_moving = {d: 0.0 for d in self.main_lanes}
            main_lane_stopped = {d: 0.0 for d in self.main_lanes}
            for direct, val in raw_moving_density.items():
                for main in self.main_lanes:
                    if main in direct: main_lane_moving[main] += val
            for direct, val in raw_stopped_density.items():
                for main in self.main_lanes:
                    if main in direct: main_lane_stopped[main] += val
            
            time_step = 0.033
            for lane in self.main_lanes:
                if main_lane_stopped[lane] > 0.5 and main_lane_stopped[lane] >= main_lane_moving[lane]:
                    self.lane_wait_timers[lane] += time_step        
                else: self.lane_wait_timers[lane] = 0.0
                
            if sum(main_lane_moving.values()) > 0: active_lane = max(main_lane_moving, key=main_lane_moving.get)
            else: active_lane = max(self.lane_wait_timers, key=self.lane_wait_timers.get)
            
            count = main_lane_moving[active_lane]
            green_time = min(120, max(10, 5 + (count * 2)))
            
            return annotated_frame, main_lane_moving, active_lane, green_time, None, self.lane_wait_timers, vehicle_counts, len(self.unique_vehicle_classes), self.violation_stats, self.recent_violations

        elif mode == "Violation":
            annotated_frame, new_evidence = self.violation_engine.detect_violations_full_frame(frame, frame_count, is_video=is_video)
            
            if new_evidence:
                self.violation_stats["Total Violations"] += len(new_evidence)
                for ev in new_evidence:
                    for v in ev['violations']:
                        self.violation_stats[v] = self.violation_stats.get(v, 0) + 1
                    self.recent_violations.append(ev)
            
            return annotated_frame, {}, "N/A", 0, None, {}, {}, 0, self.violation_stats, self.recent_violations