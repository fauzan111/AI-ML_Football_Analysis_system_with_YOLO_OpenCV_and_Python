import cv2
import numpy as np

class SpeedEstimator:
    def __init__(self, fps=24):
        self.fps = fps
        self.position_history = {} # track_id -> list of (x, y)
        self.speed_history = {} # track_id -> list of speeds
        self.frame_window = 5 # Number of frames to smooth over
        
        # Approximate pixels per meter. 
        # We will update this dynamically based on player height.
        self.pixels_per_meter = None 

    def estimate_speed(self, tracks):
        """
        tracks: dict {track_id: bbox}
        Returns: dict {track_id: speed_kmh}
        """
        current_speeds = {}
        
        # 1. Update Pixels Per Meter Estimate
        # Assumption: Average player height is 1.75m
        heights = []
        for _, bbox in tracks.items():
            h = bbox[3] - bbox[1]
            heights.append(h)
        
        if heights:
            avg_height_px = np.mean(heights)
            # Simple heuristic: Use a moving average or just current frame avg
            if self.pixels_per_meter is None:
                self.pixels_per_meter = avg_height_px / 1.75
            else:
                # Smooth update
                self.pixels_per_meter = 0.9 * self.pixels_per_meter + 0.1 * (avg_height_px / 1.75)
        
        if self.pixels_per_meter is None or self.pixels_per_meter == 0:
            return {}

        # 2. Calculate Speed
        for track_id, bbox in tracks.items():
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = bbox[3] # Use bottom center for position (feet)
            
            if track_id not in self.position_history:
                self.position_history[track_id] = []
            
            self.position_history[track_id].append((center_x, center_y))
            
            # Need at least 2 points
            if len(self.position_history[track_id]) < 2:
                continue
            
            # Keep history short
            if len(self.position_history[track_id]) > self.frame_window:
                self.position_history[track_id].pop(0)
                
            # Calculate distance moved in last step
            prev_pos = self.position_history[track_id][-2]
            curr_pos = self.position_history[track_id][-1]
            
            dist_px = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            dist_m = dist_px / self.pixels_per_meter
            
            speed_mps = dist_m * self.fps
            speed_kmh = speed_mps * 3.6
            
            # Smooth speed
            if track_id not in self.speed_history:
                self.speed_history[track_id] = []
            
            self.speed_history[track_id].append(speed_kmh)
            if len(self.speed_history[track_id]) > self.frame_window:
                self.speed_history[track_id].pop(0)
            
            avg_speed = np.mean(self.speed_history[track_id])
            current_speeds[track_id] = avg_speed
            
        return current_speeds
