from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
from utils import read_video, save_video
from team_assigner import TeamAssigner
from jersey_extractor import JerseyExtractor
from player_database import PLAYER_DATABASE
from speed_estimator import SpeedEstimator

def main():
    # 1. Load Video
    video_path = 'input.mp4'
    video_frames = read_video(video_path)
    if not video_frames:
        print("Error: Could not read video. Make sure 'input.mp4' is in the directory.")
        return
    
    height, width, _ = video_frames[0].shape

    # 2. Load Models
    model = YOLO('yolov8m.pt')
    tracker = sv.ByteTrack()
    team_assigner = TeamAssigner()
    jersey_extractor = JerseyExtractor()
    speed_estimator = SpeedEstimator()

    # 3. Initialize Annotators
    # Use EllipseAnnotator for players
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.DEFAULT,
        thickness=2,
        start_angle=-45,
        end_angle=235
    )
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_padding=10,
        text_thickness=1,
    )
    trace_annotator = sv.TraceAnnotator()

    # Storage for tracks
    tracks = {} # track_id -> {'team': int, 'number': int, 'name': str, 'history': []}
    
    # Ball Control Stats
    team_ball_control = {1: 0, 2: 0}
    
    # Output frames
    output_frames = []

    print("Starting processing...")
    
    # Initialize team colors on the first frame with detections
    team_assigned = False

    for frame_idx, frame in enumerate(video_frames):
        # Run YOLOv8
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter for 'person' (class_id 0) and 'sports ball' (class_id 32)
        player_detections = detections[detections.class_id == 0]
        ball_detections = detections[detections.class_id == 32]

        # Tracking
        player_detections = tracker.update_with_detections(player_detections)
        
        # Assign Teams (only once initially to fit KMeans)
        if not team_assigned and len(player_detections) > 0:
            # Prepare dictionary for team assigner: {track_id: bbox}
            detection_dict = {}
            for i in range(len(player_detections)):
                track_id = player_detections.tracker_id[i]
                bbox = player_detections.xyxy[i]
                detection_dict[track_id] = bbox
            
            team_assigner.assign_team_color(frame, detection_dict)
            team_assigned = True
            print("Teams assigned.")

        # Speed Estimation
        track_bboxes = {}
        for i in range(len(player_detections)):
            track_bboxes[player_detections.tracker_id[i]] = player_detections.xyxy[i]
        
        speeds = speed_estimator.estimate_speed(track_bboxes)

        # Ball Control Logic
        # Find player closest to ball
        ball_pos = None
        if len(ball_detections) > 0:
            ball_bbox = ball_detections.xyxy[0]
            ball_pos = ((ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2)
            
            min_dist = float('inf')
            controlling_team = None
            
            for i in range(len(player_detections)):
                track_id = player_detections.tracker_id[i]
                bbox = player_detections.xyxy[i]
                player_pos = ((bbox[0] + bbox[2]) / 2, bbox[3]) # Feet position
                
                dist = np.sqrt((ball_pos[0] - player_pos[0])**2 + (ball_pos[1] - player_pos[1])**2)
                
                if dist < 70: # Threshold in pixels
                    if dist < min_dist:
                        min_dist = dist
                        # Check team
                        if track_id in tracks and tracks[track_id]['team']:
                            controlling_team = tracks[track_id]['team']
            
            if controlling_team:
                team_ball_control[controlling_team] += 1

        # Process each player track
        labels = []
        for i in range(len(player_detections)):
            track_id = player_detections.tracker_id[i]
            bbox = player_detections.xyxy[i]

            if track_id not in tracks:
                tracks[track_id] = {'team': None, 'number': None, 'name': None, 'history': []}

            # Assign Team
            if team_assigned:
                team_id = team_assigner.get_player_team(frame, bbox, track_id)
                tracks[track_id]['team'] = team_id

            # Extract Jersey Number
            if tracks[track_id]['number'] is None and frame_idx % 10 == 0:
                number = jersey_extractor.extract_number(frame, bbox)
                if number is not None:
                    tracks[track_id]['number'] = number
                    team_name = f"Team {tracks[track_id]['team']}"
                    if team_name in PLAYER_DATABASE and number in PLAYER_DATABASE[team_name]:
                        tracks[track_id]['name'] = PLAYER_DATABASE[team_name][number]
            
            # Build Label
            label = f"#{track_id}"
            if tracks[track_id]['number']:
                label = f"{tracks[track_id]['number']}" # Just number is cleaner
            if tracks[track_id]['name']:
                label += f" {tracks[track_id]['name']}"
            
            # Add Speed
            if track_id in speeds:
                label += f" {speeds[track_id]:.1f} km/h"
            
            # Store Position History
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            tracks[track_id]['history'].append((center_x, center_y))

            labels.append(label)

        # Draw Annotations
        annotated_frame = frame.copy()
        
        # Draw Ellipses
        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=player_detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=player_detections,
            labels=labels
        )
        
        # Draw Ball (Red Circle)
        for bbox in ball_detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            center = ((x1+x2)//2, (y1+y2)//2)
            cv2.circle(annotated_frame, center, 5, (0, 0, 255), -1)
            # Draw triangle above ball
            pts = np.array([[center[0], center[1]-15], [center[0]-5, center[1]-25], [center[0]+5, center[1]-25]], np.int32)
            cv2.fillPoly(annotated_frame, [pts], (0, 0, 255))

        # Draw Ball Control Overlay
        total_control = team_ball_control[1] + team_ball_control[2]
        if total_control > 0:
            t1_pct = (team_ball_control[1] / total_control) * 100
            t2_pct = (team_ball_control[2] / total_control) * 100
        else:
            t1_pct = 50
            t2_pct = 50
            
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (width-350, height-100), (width-50, height-20), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        
        cv2.putText(annotated_frame, f"Team 1 Possession: {t1_pct:.1f}%", (width-340, height-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(annotated_frame, f"Team 2 Possession: {t2_pct:.1f}%", (width-340, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        output_frames.append(annotated_frame)
        
        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}/{len(video_frames)}")

    # Save Output
    print("Saving output video...")
    save_video(output_frames, 'output_video.avi')
    
    # Save Tracks for Heatmap
    import pickle
    with open('tracks.pkl', 'wb') as f:
        pickle.dump(tracks, f)
    print("Tracks saved to tracks.pkl")
    
    print("Done!")

if __name__ == '__main__':
    main()
