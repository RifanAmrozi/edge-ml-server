import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import math
import json
import os
from datetime import datetime
from enum import Enum


class DetectionPhase(Enum):
    """Fase deteksi shoplifting"""
    IDLE = "idle"
    REACHING_SHELF = "reaching_shelf"
    GRABBING = "grabbing"
    SUSPICIOUS_MOVEMENT = "suspicious"
    ALERT = "alert"


class SuspiciousPose(Enum):
    BENDING_DOWN = "bending_down"
    CROUCHING = "crouching"
    HIDING_UNDER_CLOTHING = "hiding_under_clothing"
    CONCEALING_AT_WAIST = "concealing_at_waist"
    REACHING_POCKET = "reaching_pocket"
    HANDS_NEAR_BODY = "hands_near_body"
    PUTTING_IN_PANTS_POCKET = "putting_in_pants_pocket"
    HANDS_BEHIND_BACK = "hands_behind_back"
    SQUATTING_LOW = "squatting_low"
    REACHING_WAIST_BACK = "reaching_waist_back"
    ZONE_PANTS_POCKET_LEFT = "zone_pants_pocket_left"
    ZONE_PANTS_POCKET_RIGHT = "zone_pants_pocket_right"
    ZONE_JACKET_POCKET_LEFT = "zone_jacket_pocket_left"
    ZONE_JACKET_POCKET_RIGHT = "zone_jacket_pocket_right"


class PocketZone:
    
    def __init__(self, zone_type, left_point, right_point, width_factor=0.35, depth_factor=0.3):
        self.zone_type = zone_type
        self.left_point = left_point
        self.right_point = right_point
        self.width_factor = width_factor
        self.depth_factor = depth_factor
        self.zone_box = None
    
    def calculate_zone(self, shoulder_width):
        """Hitung bounding box zona kantong"""
        if not self.left_point or not self.right_point:
            return None
        
        zone_width = shoulder_width * self.width_factor
        
        if 'pants' in self.zone_type:
            zone_height = shoulder_width * 0.5        
            x_center = (self.left_point[0] + self.right_point[0]) / 2
            y_center = (self.left_point[1] + self.right_point[1]) / 2 - shoulder_width * 0.2 
        else:
            zone_height = shoulder_width * 0.5       
            x_center = (self.left_point[0] + self.right_point[0]) / 2
            y_center = (self.left_point[1] + self.right_point[1]) / 2 + shoulder_width * 0.15  
        
        if 'left' in self.zone_type:
            x1 = x_center - zone_width * 1.0      
            x2 = x_center + zone_width * 0.6      
        else:
            x1 = x_center - zone_width * 0.6     
            x2 = x_center + zone_width * 1.0     
        
        y1 = y_center
        y2 = y_center + zone_height
        
        self.zone_box = (int(x1), int(y1), int(x2), int(y2))
        return self.zone_box
    
    def is_point_in_zone(self, point):
        """Cek apakah point masuk ke zona"""
        if not self.zone_box or not point:
            return False
        
        x1, y1, x2, y2 = self.zone_box
        x, y = point[0], point[1]
        
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def get_penetration_depth(self, point):
        """Hitung seberapa dalam tangan masuk ke zona (0-1)"""
        if not self.zone_box or not point:
            return 0
        
        x1, y1, x2, y2 = self.zone_box
        x, y = point[0], point[1]
        
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return 0
        
        if 'left' in self.zone_type:
            depth = (x - x1) / (x2 - x1) if (x2 - x1) > 0 else 0
        else:
            depth = (x2 - x) / (x2 - x1) if (x2 - x1) > 0 else 0
        
        return depth


class ShopliftingPoseDetectorWithGrab:
    def __init__(self, pose_model="yolo11m-pose.pt", debug_mode=False):
        print("🚀 Initializing Shoplifting Detector...")
        print("   [SHOPLIFTING DETECTION]   ")
        
        try:
            self.pose_model = YOLO(pose_model)
            print("✅ Pose detection model loaded")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        self.frame_count = 0
        self.debug_mode = debug_mode
        
        self.KEYPOINTS = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
        self.person_tracks = defaultdict(lambda: {
            'phase': DetectionPhase.IDLE,
            'phase_start_frame': 0,
            
            # Grabbing detection
            'wrist_positions': deque(maxlen=20),
            'hand_extended': False,
            'hand_extended_frames': 0,
            'grab_detected': False,
            'grab_frame': 0,
            'grabbed_hand': None,
            
            # Suspicious pose tracking
            'suspicious_poses': deque(maxlen=30),
            'pose_counts': defaultdict(int),
            'alert_triggered': False,
            'last_alert_frame': 0,
            'suspicion_score': 0.0,
            'consecutive_suspicious': 0,
            'total_frames_tracked': 0,
            'suspicious_ratio': 0.0,
            'first_seen': time.time(),

            # Suspicious validation
            'suspicious_buffer': deque(maxlen=15),
            'suspicious_frame_count': 0,
            'last_normal_frame': 0,
            
            'zone_penetration_detected': False,
            'zone_penetration_frames': 0,
            'zone_penetration_zones': [],

            # History
            'phase_history': [],
            
            # Pocket zone tracking
            'pocket_zones': {},
            'wrist_in_zone_frames': defaultdict(int),
            'zone_entry_frames': defaultdict(int),
            'max_zone_depth': defaultdict(float),
            'zone_detections': defaultdict(list),
            'current_keypoints': None,
        })
        
        # THRESHOLDS - Grabbing phase
        self.GRAB_THRESHOLDS = {
            'hand_extension_threshold': 100,
            'hand_height_tolerance': 150,
            'min_extension_frames': 5,
            'grab_timeout': 90,
            'hand_close_distance': 60,
            'elbow_angle_extended': 120,
            'elbow_angle_grab': 140,
            'distance_reduction_threshold': 30,
            'velocity_threshold': 8,
        }

        self.SUSPICIOUS_VALIDATION = {
            'min_suspicious_frames': 8,
            'suspicious_confidence_threshold': 0.75,
            'pose_consistency_window': 15,
            'min_unique_poses': 2,
            'high_severity_poses': [
                SuspiciousPose.HIDING_UNDER_CLOTHING,
                SuspiciousPose.PUTTING_IN_PANTS_POCKET,
                SuspiciousPose.ZONE_PANTS_POCKET_LEFT,
                SuspiciousPose.ZONE_PANTS_POCKET_RIGHT,
                SuspiciousPose.ZONE_JACKET_POCKET_LEFT,
                SuspiciousPose.ZONE_JACKET_POCKET_RIGHT

            ],
            'timeout_normal_behavior': 60
        }

        self.SUSPICIOUS_THRESHOLDS = {
            'bending_threshold': 0.55,
            'crouch_knee_angle': 110,
            'waist_distance_threshold': 70,
            'suspicious_frame_count': 10,
            'alert_cooldown': 90,
            'high_confidence_threshold': 0.90,
            'score_threshold': 75.0,
            'score_decay': 3.0,
            'continuous_pose_bonus': 15,
            'min_consecutive_for_bonus': 12,
            'min_tracking_frames': 15,
            'suspicious_ratio_threshold': 0.35
        }
        
        # Zone thresholds
        self.ZONE_THRESHOLDS = {
            'min_frames_in_zone': 2,              
            'min_penetration_depth': 0.20,        
            'high_confidence_depth': 0.40,         
            'zone_based_alert_score': 30,
            'immediate_suspicious_depth': 0.30,   
            'immediate_suspicious_frames': 2,      
        }
        
        self.alert_log = []
        self.session_start = datetime.now()
        self.frame_buffer = deque(maxlen=450)
        self.alert_clips_saved = []
        self.fps = 30
        
        self.recording_alerts = {}
        
        print("✅ Initialization complete")
    
    def get_keypoint(self, keypoints, name):
        """Get keypoint by name dengan confidence check"""
        idx = self.KEYPOINTS[name]
        if idx < len(keypoints):
            x, y, conf = keypoints[idx]
            return (float(x), float(y), float(conf)) if conf > 0.6 else None
        return None
    
    def calculate_angle(self, p1, p2, p3):
        """Hitung sudut antara 3 titik"""
        if not all([p1, p2, p3]):
            return None
        
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def distance(self, p1, p2):
        """Euclidean distance"""
        if not all([p1, p2]):
            return None
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def initialize_pocket_zones(self, track_id, keypoints):
        """Inisialisasi zona kantong untuk person baru"""
        track = self.person_tracks[track_id]
        
        if track['pocket_zones']:
            return
        
        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
        left_hip = self.get_keypoint(keypoints, 'left_hip')
        right_hip = self.get_keypoint(keypoints, 'right_hip')
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return
        
        shoulder_width = self.distance(left_shoulder, right_shoulder)
        if not shoulder_width:
            return
        
        track['pocket_zones'] = {
            'pants_pocket_left': PocketZone(
                'pants_pocket_left',
                left_hip, left_hip,
                width_factor=0.4,      
                depth_factor=0.35
            ),
            'pants_pocket_right': PocketZone(
                'pants_pocket_right',
                right_hip, right_hip,
                width_factor=0.4,      
                depth_factor=0.35
            ),
            'jacket_pocket_left': PocketZone(
                'jacket_pocket_left',
                left_shoulder, left_hip,
                width_factor=0.35,    
                depth_factor=0.3
            ),
            'jacket_pocket_right': PocketZone(
                'jacket_pocket_right',
                right_shoulder, right_hip,
                width_factor=0.35,     
                depth_factor=0.3
            )
        }
        
        if self.debug_mode:
            print(f"  Initialize pocket zones for Track {track_id} (shoulder_width: {shoulder_width:.0f}px)")
    
    def update_pocket_zones(self, track_id, keypoints):
        """Update zona kantong berdasarkan posisi body terbaru"""
        track = self.person_tracks[track_id]
        
        if not track['pocket_zones']:
            return
        
        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
        left_hip = self.get_keypoint(keypoints, 'left_hip')
        right_hip = self.get_keypoint(keypoints, 'right_hip')
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return
        
        shoulder_width = self.distance(left_shoulder, right_shoulder)
        if not shoulder_width:
            return
        
        track['pocket_zones']['pants_pocket_left'].left_point = left_hip
        track['pocket_zones']['pants_pocket_left'].right_point = left_hip
        track['pocket_zones']['pants_pocket_left'].calculate_zone(shoulder_width)
        
        track['pocket_zones']['pants_pocket_right'].left_point = right_hip
        track['pocket_zones']['pants_pocket_right'].right_point = right_hip
        track['pocket_zones']['pants_pocket_right'].calculate_zone(shoulder_width)
        
        track['pocket_zones']['jacket_pocket_left'].left_point = left_shoulder
        track['pocket_zones']['jacket_pocket_left'].right_point = left_hip
        track['pocket_zones']['jacket_pocket_left'].calculate_zone(shoulder_width)
        
        track['pocket_zones']['jacket_pocket_right'].left_point = right_shoulder
        track['pocket_zones']['jacket_pocket_right'].right_point = right_hip
        track['pocket_zones']['jacket_pocket_right'].calculate_zone(shoulder_width)
    
    def detect_zone_penetration(self, track_id, keypoints):
        """
        DETEKSI ZONA: Cek apakah tangan masuk ke zona kantong setelah grab
        Returns: (has_zone_penetration, zone_details, confidence)
        """
        track = self.person_tracks[track_id]
        
        zone_poses = [] 
            
        zone_to_pose = {
                'pants_pocket_left': SuspiciousPose.ZONE_PANTS_POCKET_LEFT,
                'pants_pocket_right': SuspiciousPose.ZONE_PANTS_POCKET_RIGHT,
                'jacket_pocket_left': SuspiciousPose.ZONE_JACKET_POCKET_LEFT,
                'jacket_pocket_right': SuspiciousPose.ZONE_JACKET_POCKET_RIGHT
        }

        if not track['grab_detected']:
            return False, [], 0.0
        
        if not track['pocket_zones']:
            self.initialize_pocket_zones(track_id, keypoints)
            return False, [], 0.0
        
        self.update_pocket_zones(track_id, keypoints)
        
        left_wrist = self.get_keypoint(keypoints, 'left_wrist')
        right_wrist = self.get_keypoint(keypoints, 'right_wrist')
        
        zone_details = []
        confidence = 0.0
        
        if left_wrist:
            for zone_name, zone in track['pocket_zones'].items():
                if zone.is_point_in_zone(left_wrist):
                    depth = zone.get_penetration_depth(left_wrist)
                    
                    track['wrist_in_zone_frames'][zone_name] += 1
                    track['max_zone_depth'][zone_name] = max(
                        track['max_zone_depth'][zone_name],
                        depth
                    )
                    
                    if track['wrist_in_zone_frames'][zone_name] >= self.ZONE_THRESHOLDS['min_frames_in_zone']:
                        depth_conf = min(1.0, depth / self.ZONE_THRESHOLDS['high_confidence_depth'])
                        frame_conf = min(1.0, track['wrist_in_zone_frames'][zone_name] / 10)
                        zone_conf = (depth_conf + frame_conf) / 2 * 0.95
                        
                        zone_details.append({
                            'zone': zone_name,
                            'hand': 'left',
                            'depth': depth,
                            'frames_in_zone': track['wrist_in_zone_frames'][zone_name],
                            'confidence': zone_conf,
                            'severity': 'high' if depth > self.ZONE_THRESHOLDS['high_confidence_depth'] else 'medium'
                        })

                        pose_type = zone_to_pose[zone_name]
                        zone_poses.append((
                            pose_type,
                            zone_conf,
                            f"Tangan kiri masuk {zone_name.replace('_', ' ')} (depth: {depth:.1%})"
                        ))
                        
                        confidence = max(confidence, zone_conf)
                else:
                    if track['wrist_in_zone_frames'][zone_name] > 0:
                        if track['max_zone_depth'][zone_name] > self.ZONE_THRESHOLDS['min_penetration_depth']:
                            track['zone_detections'][zone_name].append({
                                'frame': self.frame_count,
                                'max_depth': track['max_zone_depth'][zone_name],
                                'frames_in_zone': track['wrist_in_zone_frames'][zone_name],
                                'hand': 'left'
                            })
                        
                        track['wrist_in_zone_frames'][zone_name] = 0
                        track['max_zone_depth'][zone_name] = 0
        
        if right_wrist:
            for zone_name, zone in track['pocket_zones'].items():
                if zone.is_point_in_zone(right_wrist):
                    depth = zone.get_penetration_depth(right_wrist)
                    
                    track['wrist_in_zone_frames'][zone_name] += 1
                    track['max_zone_depth'][zone_name] = max(
                        track['max_zone_depth'][zone_name],
                        depth
                    )
                    
                    if track['wrist_in_zone_frames'][zone_name] >= self.ZONE_THRESHOLDS['min_frames_in_zone']:
                        depth_conf = min(1.0, depth / self.ZONE_THRESHOLDS['high_confidence_depth'])
                        frame_conf = min(1.0, track['wrist_in_zone_frames'][zone_name] / 10)
                        zone_conf = (depth_conf + frame_conf) / 2 * 0.95
                        
                        zone_details.append({
                            'zone': zone_name,
                            'hand': 'right',
                            'depth': depth,
                            'frames_in_zone': track['wrist_in_zone_frames'][zone_name],
                            'confidence': zone_conf,
                            'severity': 'high' if depth > self.ZONE_THRESHOLDS['high_confidence_depth'] else 'medium'
                        })

                        pose_type = zone_to_pose[zone_name]
                        zone_poses.append((
                            pose_type,
                            zone_conf,
                            f"Tangan kanan masuk {zone_name.replace('_', ' ')} (depth: {depth:.1%})"
                        ))
                        
                        confidence = max(confidence, zone_conf)
                else:
                    if track['wrist_in_zone_frames'][zone_name] > 0:
                        if track['max_zone_depth'][zone_name] > self.ZONE_THRESHOLDS['min_penetration_depth']:
                            track['zone_detections'][zone_name].append({
                                'frame': self.frame_count,
                                'max_depth': track['max_zone_depth'][zone_name],
                                'frames_in_zone': track['wrist_in_zone_frames'][zone_name],
                                'hand': 'right'
                            })
                        
                        track['wrist_in_zone_frames'][zone_name] = 0
                        track['max_zone_depth'][zone_name] = 0
        
        has_penetration = len(zone_details) > 0
        return has_penetration, zone_details, confidence, zone_poses
    
    def detect_hand_reaching(self, keypoints, track_id):
        """FASE 1: Deteksi tangan meraih"""
        track = self.person_tracks[track_id]
        
        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
        left_elbow = self.get_keypoint(keypoints, 'left_elbow')
        right_elbow = self.get_keypoint(keypoints, 'right_elbow')
        left_wrist = self.get_keypoint(keypoints, 'left_wrist')
        right_wrist = self.get_keypoint(keypoints, 'right_wrist')
        left_hip = self.get_keypoint(keypoints, 'left_hip')
        right_hip = self.get_keypoint(keypoints, 'right_hip')
        
        if left_wrist:
            track['wrist_positions'].append(('left', left_wrist, self.frame_count))
        if right_wrist:
            track['wrist_positions'].append(('right', right_wrist, self.frame_count))
        
        reaching_detected = False
        hand_side = None
        confidence = 0.0
        
        if all([left_shoulder, left_elbow, left_wrist, left_hip]):
            wrist_to_shoulder_dist = self.distance(left_wrist, left_shoulder)
            elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            if wrist_to_shoulder_dist and wrist_to_shoulder_dist > self.GRAB_THRESHOLDS['hand_extension_threshold']:
                if elbow_angle and elbow_angle > self.GRAB_THRESHOLDS['elbow_angle_extended']:
                    height_diff = left_wrist[1] - left_shoulder[1]
                    wrist_to_hip_dist = self.distance(left_wrist, left_hip)
                    
                    if height_diff < self.GRAB_THRESHOLDS['hand_height_tolerance']:
                        if wrist_to_hip_dist and wrist_to_hip_dist > 80:
                            reaching_detected = True
                            hand_side = 'left'
                            
                            dist_conf = min(1.0, wrist_to_shoulder_dist / 200)
                            angle_conf = min(1.0, elbow_angle / 180)
                            height_conf = 1.0 - (abs(height_diff) / self.GRAB_THRESHOLDS['hand_height_tolerance'])
                            
                            confidence = (dist_conf + angle_conf + height_conf) / 3 * 0.95
        
        if all([right_shoulder, right_elbow, right_wrist, right_hip]) and not reaching_detected:
            wrist_to_shoulder_dist = self.distance(right_wrist, right_shoulder)
            elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            if wrist_to_shoulder_dist and wrist_to_shoulder_dist > self.GRAB_THRESHOLDS['hand_extension_threshold']:
                if elbow_angle and elbow_angle > self.GRAB_THRESHOLDS['elbow_angle_extended']:
                    height_diff = right_wrist[1] - right_shoulder[1]
                    wrist_to_hip_dist = self.distance(right_wrist, right_hip)
                    
                    if height_diff < self.GRAB_THRESHOLDS['hand_height_tolerance']:
                        if wrist_to_hip_dist and wrist_to_hip_dist > 80:
                            reaching_detected = True
                            hand_side = 'right'
                            
                            dist_conf = min(1.0, wrist_to_shoulder_dist / 200)
                            angle_conf = min(1.0, elbow_angle / 180)
                            height_conf = 1.0 - (abs(height_diff) / self.GRAB_THRESHOLDS['hand_height_tolerance'])
                            
                            confidence = (dist_conf + angle_conf + height_conf) / 3 * 0.95
        
        return reaching_detected, hand_side, confidence
    
    def detect_grabbing_motion(self, keypoints, track_id):
        """FASE 2: Deteksi gerakan menggenggam"""
        track = self.person_tracks[track_id]
        
        if not track['hand_extended']:
            return False, 0.0
        
        grabbed_hand = track['grabbed_hand']
        if not grabbed_hand:
            return False, 0.0
        
        shoulder_key = f'{grabbed_hand}_shoulder'
        wrist_key = f'{grabbed_hand}_wrist'
        elbow_key = f'{grabbed_hand}_elbow'
        hip_key = f'{grabbed_hand}_hip'
        
        shoulder = self.get_keypoint(keypoints, shoulder_key)
        wrist = self.get_keypoint(keypoints, wrist_key)
        elbow = self.get_keypoint(keypoints, elbow_key)
        hip = self.get_keypoint(keypoints, hip_key)
        
        if not all([shoulder, wrist, elbow, hip]):
            return False, 0.0
        
        current_dist_shoulder = self.distance(wrist, shoulder)
        current_dist_hip = self.distance(wrist, hip)
        
        grab_detected = False
        confidence = 0.0
        
        if len(track['wrist_positions']) >= 5:
            past_positions = [p for p in track['wrist_positions'] 
                            if p[0] == grabbed_hand and 
                            self.frame_count - p[2] >= 3 and 
                            self.frame_count - p[2] <= 6]
            
            if past_positions:
                past_wrist = past_positions[0][1]
                past_dist_shoulder = self.distance(past_wrist, shoulder)
                past_dist_hip = self.distance(past_wrist, hip)
                
                frame_diff = self.frame_count - past_positions[0][2]
                wrist_movement = self.distance(wrist, past_wrist)
                velocity = wrist_movement / max(frame_diff, 1) if wrist_movement else 0
                
                conditions_met = 0
                total_confidence = 0.0
                
                if past_dist_shoulder and current_dist_shoulder:
                    distance_reduction_shoulder = past_dist_shoulder - current_dist_shoulder
                    if distance_reduction_shoulder > self.GRAB_THRESHOLDS['distance_reduction_threshold']:
                        conditions_met += 1
                        total_confidence += min(0.35, distance_reduction_shoulder / 100)
                
                if past_dist_hip and current_dist_hip:
                    distance_reduction_hip = past_dist_hip - current_dist_hip
                    if distance_reduction_hip > self.GRAB_THRESHOLDS['distance_reduction_threshold']:
                        conditions_met += 1
                        total_confidence += min(0.35, distance_reduction_hip / 100)
                
                if velocity > self.GRAB_THRESHOLDS['velocity_threshold']:
                    conditions_met += 1
                    total_confidence += min(0.30, velocity / 20)
                
                elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                if elbow_angle and elbow_angle < self.GRAB_THRESHOLDS['elbow_angle_grab']:
                    conditions_met += 1
                    total_confidence += 0.30
                
                y_movement = wrist[1] - past_wrist[1]
                if y_movement > 20:
                    conditions_met += 1
                    total_confidence += 0.25
                
                if conditions_met >= 2:
                    grab_detected = True
                    confidence = min(0.95, total_confidence)
                    
                    if self.debug_mode:
                        print(f"  GRAB: {conditions_met}/4 conditions | conf: {confidence:.2f}")
        
        return grab_detected, confidence
    
    def detect_suspicious_poses(self, keypoints):
        """FASE 3: Deteksi pose mencurigakan"""
        suspicious_poses = []
        
        nose = self.get_keypoint(keypoints, 'nose')
        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
        left_elbow = self.get_keypoint(keypoints, 'left_elbow')
        right_elbow = self.get_keypoint(keypoints, 'right_elbow')
        left_wrist = self.get_keypoint(keypoints, 'left_wrist')
        right_wrist = self.get_keypoint(keypoints, 'right_wrist')
        left_hip = self.get_keypoint(keypoints, 'left_hip')
        right_hip = self.get_keypoint(keypoints, 'right_hip')
        left_knee = self.get_keypoint(keypoints, 'left_knee')
        right_knee = self.get_keypoint(keypoints, 'right_knee')
        left_ankle = self.get_keypoint(keypoints, 'left_ankle')
        right_ankle = self.get_keypoint(keypoints, 'right_ankle')
        
        # 1. BENDING DOWN
        if all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2
            torso_height = abs(hip_y - shoulder_y)
            nose_to_hip = abs(nose[1] - hip_y)
            bend_ratio = nose_to_hip / (torso_height + 1e-6)
            
            if bend_ratio < self.SUSPICIOUS_THRESHOLDS['bending_threshold']:
                confidence = (1.0 - bend_ratio) * 0.90
                suspicious_poses.append((
                    SuspiciousPose.BENDING_DOWN,
                    confidence,
                    f"Membungkuk setelah ambil barang"
                ))
        
        # 2. SQUATTING/CROUCHING
        if all([left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
            left_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            if left_angle and right_angle:
                avg_angle = (left_angle + right_angle) / 2
                
                if avg_angle < 95:
                    confidence = (1.0 - (avg_angle / 95)) * 0.92
                    suspicious_poses.append((
                        SuspiciousPose.SQUATTING_LOW,
                        confidence,
                        f"Jongkok dengan barang (sudut: {avg_angle:.0f}°)"
                    ))
                elif avg_angle < self.SUSPICIOUS_THRESHOLDS['crouch_knee_angle']:
                    confidence = (1.0 - (avg_angle / self.SUSPICIOUS_THRESHOLDS['crouch_knee_angle'])) * 0.85
                    suspicious_poses.append((
                        SuspiciousPose.CROUCHING,
                        confidence,
                        f"Berjongkok dengan barang"
                    ))
        
        # 3. HIDING UNDER CLOTHING
        if all([left_wrist, right_wrist, left_shoulder, right_shoulder, left_hip, right_hip]):
            chest_y = (left_shoulder[1] + right_shoulder[1]) / 2
            belly_y = (left_hip[1] + right_hip[1]) / 2
            
            left_at_torso = chest_y < left_wrist[1] < belly_y
            right_at_torso = chest_y < right_wrist[1] < belly_y
            
            torso_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            left_dist = abs(left_wrist[0] - torso_center_x)
            right_dist = abs(right_wrist[0] - torso_center_x)
            torso_width = abs(left_shoulder[0] - right_shoulder[0])
            
            if left_at_torso and right_at_torso:
                if left_dist < torso_width * 0.45 and right_dist < torso_width * 0.45:
                    confidence = 0.95
                    suspicious_poses.append((
                        SuspiciousPose.HIDING_UNDER_CLOTHING,
                        confidence,
                        "🚨 Memasukkan barang ke baju"
                    ))
            elif left_at_torso or right_at_torso:
                active_dist = left_dist if left_at_torso else right_dist
                if active_dist < torso_width * 0.35:
                    confidence = 0.90
                    suspicious_poses.append((
                        SuspiciousPose.HIDING_UNDER_CLOTHING,
                        confidence,
                        "⚠️ Tangan masuk ke baju"
                    ))
        
        # 4. PUTTING IN PANTS POCKET - Kiri
        if all([left_shoulder, left_elbow, left_wrist, left_hip, left_knee]):
            left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            if left_wrist[1] > left_hip[1] and left_wrist[1] < left_knee[1]:
                left_to_hip_x = abs(left_wrist[0] - left_hip[0])
                left_to_hip_y = abs(left_wrist[1] - left_hip[1])
                
                if left_elbow_angle and left_elbow_angle < 125:
                    if left_to_hip_x < 70 and left_to_hip_y < 100:
                        if left_wrist[2] < 0.75:
                            confidence = 0.88
                            suspicious_poses.append((
                                SuspiciousPose.PUTTING_IN_PANTS_POCKET,
                                confidence,
                                "🚨 Memasukkan ke kantong celana"
                            ))
                        elif left_to_hip_x < 30:
                            confidence = 0.85
                            suspicious_poses.append((
                                SuspiciousPose.PUTTING_IN_PANTS_POCKET,
                                confidence,
                                "🚨 Memasukkan ke kantong celana"
                            ))
        
        # 5. PUTTING IN PANTS POCKET - Kanan
        if all([right_shoulder, right_elbow, right_wrist, right_hip, right_knee]):
            right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            if right_wrist[1] > right_hip[1] and right_wrist[1] < right_knee[1]:
                right_to_hip_x = abs(right_wrist[0] - right_hip[0])
                right_to_hip_y = abs(right_wrist[1] - right_hip[1])
                
                if right_elbow_angle and right_elbow_angle < 125:
                    if right_to_hip_x < 70 and right_to_hip_y < 110:
                        confidence = 0.88
                        suspicious_poses.append((
                            SuspiciousPose.PUTTING_IN_PANTS_POCKET,
                            confidence,
                            "🚨 Memasukkan ke kantong celana"
                        ))
        
        # 6. CONCEALING AT WAIST
        if all([left_wrist, left_hip, left_shoulder]):
            wrist_to_hip = self.distance(left_wrist, left_hip)
            if wrist_to_hip and wrist_to_hip < self.SUSPICIOUS_THRESHOLDS['waist_distance_threshold']:
                if abs(left_wrist[0] - left_hip[0]) > 25:
                    confidence = 0.80
                    suspicious_poses.append((
                        SuspiciousPose.CONCEALING_AT_WAIST,
                        confidence,
                        "Menyembunyikan di pinggang"
                    ))
        
        if all([right_wrist, right_hip, right_shoulder]):
            wrist_to_hip = self.distance(right_wrist, right_hip)
            if wrist_to_hip and wrist_to_hip < self.SUSPICIOUS_THRESHOLDS['waist_distance_threshold']:
                if abs(right_wrist[0] - right_hip[0]) > 25:
                    confidence = 0.80
                    suspicious_poses.append((
                        SuspiciousPose.CONCEALING_AT_WAIST,
                        confidence,
                        "Menyembunyikan di pinggang"
                    ))
        
        # 7. REACHING WAIST BACK
        if all([left_shoulder, left_hip, left_wrist]):
            if left_wrist[2] < 0.65:
                wrist_to_hip = self.distance(left_wrist, left_hip)
                if wrist_to_hip and wrist_to_hip < 110:
                    confidence = 0.82
                    suspicious_poses.append((
                        SuspiciousPose.REACHING_WAIST_BACK,
                        confidence,
                        "Tangan ke belakang pinggang"
                    ))
        
        if all([right_shoulder, right_hip, right_wrist]):
            if right_wrist[2] < 0.65:
                wrist_to_hip = self.distance(right_wrist, right_hip)
                if wrist_to_hip and wrist_to_hip < 110:
                    confidence = 0.82
                    suspicious_poses.append((
                        SuspiciousPose.REACHING_WAIST_BACK,
                        confidence,
                        "Tangan ke belakang pinggang"
                    ))
        
        return suspicious_poses
    
    def update_phase(self, track_id, keypoints, current_frame):
        """Update fase deteksi dengan zone detection - FIXED LOGIC"""
        track = self.person_tracks[track_id]
        current_phase = track['phase']
        
        # FASE 1: IDLE -> REACHING_SHELF
        if current_phase == DetectionPhase.IDLE:
            is_reaching, hand_side, confidence = self.detect_hand_reaching(keypoints, track_id)
            
            if is_reaching:
                track['hand_extended'] = True
                track['hand_extended_frames'] += 1
                track['grabbed_hand'] = hand_side
                
                if track['hand_extended_frames'] >= self.GRAB_THRESHOLDS['min_extension_frames']:
                    track['phase'] = DetectionPhase.REACHING_SHELF
                    track['phase_start_frame'] = current_frame
                    
                    if self.debug_mode:
                        print(f"🟡 Track {track_id}: IDLE -> REACHING ({hand_side} hand)")
            else:
                track['hand_extended_frames'] = 0
                track['hand_extended'] = False
        
        # FASE 2: REACHING_SHELF -> GRABBING
        elif current_phase == DetectionPhase.REACHING_SHELF:
            is_grabbing, confidence = self.detect_grabbing_motion(keypoints, track_id)
            
            if is_grabbing:
                track['phase'] = DetectionPhase.GRABBING
                track['grab_detected'] = True
                track['grab_frame'] = current_frame
                track['phase_start_frame'] = current_frame
                
                if self.debug_mode:
                    print(f"🟠 Track {track_id}: REACHING -> GRABBING (confidence: {confidence:.2f})")
                return False, [], []
            else:
                is_still_reaching, _, _ = self.detect_hand_reaching(keypoints, track_id)
                
                if not is_still_reaching:
                    grabbed_hand = track['grabbed_hand']
                    wrist_key = f'{grabbed_hand}_wrist'
                    shoulder_key = f'{grabbed_hand}_shoulder'
                    hip_key = f'{grabbed_hand}_hip'
                    
                    wrist = self.get_keypoint(keypoints, wrist_key)
                    shoulder = self.get_keypoint(keypoints, shoulder_key)
                    hip = self.get_keypoint(keypoints, hip_key)
                    
                    if all([wrist, shoulder, hip]):
                        wrist_to_shoulder = self.distance(wrist, shoulder)
                        wrist_to_hip = self.distance(wrist, hip)
                        
                        if (wrist_to_shoulder and wrist_to_shoulder < 120) or \
                           (wrist_to_hip and wrist_to_hip < 100):
                            track['phase'] = DetectionPhase.GRABBING
                            track['grab_detected'] = True
                            track['grab_frame'] = current_frame
                            track['phase_start_frame'] = current_frame
                            
                            if self.debug_mode:
                                print(f"🟠 Track {track_id}: REACHING -> GRABBING (implicit)")
                            return False, [], []
                        else:
                            if current_frame - track['phase_start_frame'] > 30:
                                track['phase'] = DetectionPhase.IDLE
                                track['hand_extended'] = False
                                track['hand_extended_frames'] = 0
                                if self.debug_mode:
                                    print(f"⚪ Track {track_id}: REACHING timeout -> IDLE")
                else:
                    if current_frame - track['phase_start_frame'] > 60:
                        track['phase'] = DetectionPhase.IDLE
                        track['hand_extended'] = False
                        track['hand_extended_frames'] = 0
                        if self.debug_mode:
                            print(f"⚪ Track {track_id}: REACHING timeout -> IDLE")
        
        # FASE 3: GRABBING 
        elif current_phase == DetectionPhase.GRABBING:
            # Initialize zones
            if not track['pocket_zones']:
                self.initialize_pocket_zones(track_id, keypoints)
            
            # CEK ZONE DETECTION
            zone_penetration, zone_details, zone_conf, zone_poses = self.detect_zone_penetration(track_id, keypoints)
            
            # Deteksi suspicious poses 
            suspicious_poses = self.detect_suspicious_poses(keypoints)
            
            # Update buffer 
            track['suspicious_buffer'].append({
                'frame': current_frame,
                'has_suspicious': len(suspicious_poses) > 0 or zone_penetration,
                'poses': suspicious_poses + zone_poses,
                'zone_details': zone_details,
                'high_severity': any(p[0] in self.SUSPICIOUS_VALIDATION['high_severity_poses'] 
                                for p in suspicious_poses) or zone_penetration
            })
            
            # Flag untuk tracking zone penetration
            transition_to_suspicious = False
            transition_reason = ""
            
            if zone_penetration and zone_details:
                very_deep_zones = [z for z in zone_details 
                                  if z['depth'] >= 0.60  
                                  and z['frames_in_zone'] >= 3]  
                
                if very_deep_zones:
                    track['phase'] = DetectionPhase.ALERT
                    track['phase_start_frame'] = current_frame
                    
                    track['suspicion_score'] += 60
                    track['zone_penetration_detected'] = True
                    track['zone_penetration_frames'] = current_frame
                    track['zone_penetration_zones'] = [z['zone'] for z in very_deep_zones]
                    
                    for zone_pose in zone_poses:
                        track['pose_counts'][zone_pose[0]] += 10
                    
                    alert_reasons = []
                    for z in very_deep_zones:
                        zone_name = z['zone'].replace('_', ' ').upper()
                        alert_reasons.append(f"🚨 DEEP {zone_name} ({z['depth']:.0%})")
                    
                    if self.debug_mode:
                        print(f"🚨🚨 Track {track_id}: INSTANT ALERT - VERY DEEP ZONE PENETRATION!")
                        for z in very_deep_zones:
                            print(f"    - {z['zone']}: depth {z['depth']:.1%}, {z['frames_in_zone']}f")
                    
                    return True, suspicious_poses + zone_poses, alert_reasons
            
            # CEK 1: IMMEDIATE HIGH SEVERITY ZONE 
            if zone_penetration:
                high_severity_zones = [z for z in zone_details 
                                    if z['depth'] >= self.ZONE_THRESHOLDS['immediate_suspicious_depth']
                                    and z['frames_in_zone'] >= self.ZONE_THRESHOLDS['immediate_suspicious_frames']]
                
                if high_severity_zones:
                    # Masuk SUSPICIOUS 
                    transition_to_suspicious = True
                    transition_reason = f"ZONE_PENETRATION: {[z['zone'] for z in high_severity_zones]}"
                    
                    # Boost score
                    track['suspicion_score'] += 35
                    track['consecutive_suspicious'] = 10
                    
                    # Tambahkan zone poses
                    for zone_pose in zone_poses:
                        track['pose_counts'][zone_pose[0]] += 1
                    
                    # Set zone flags
                    track['zone_penetration_detected'] = True
                    track['zone_penetration_frames'] = current_frame
                    track['zone_penetration_zones'] = [z['zone'] for z in high_severity_zones]
                
                # CEK MEDIUM SEVERITY 
                else:
                    if 'zone_consecutive_frames' not in track:
                        track['zone_consecutive_frames'] = 0
                    
                    track['zone_consecutive_frames'] += 1
                    
                    if track['zone_consecutive_frames'] >= 5:
                        transition_to_suspicious = True
                        transition_reason = f"ACCUMULATED_ZONE: {track['zone_consecutive_frames']}f"
                        
                        track['zone_penetration_detected'] = True
                        track['zone_penetration_frames'] = current_frame
                        track['zone_penetration_zones'] = [z['zone'] for z in zone_details]
                        
                        for zone_pose in zone_poses:
                            track['pose_counts'][zone_pose[0]] += 1
            else:
                track['zone_consecutive_frames'] = 0
            
            # CEK 2: HIGH SEVERITY POSES
            if not transition_to_suspicious and suspicious_poses:
                high_severity = [p for p in suspicious_poses 
                            if p[0] in self.SUSPICIOUS_VALIDATION['high_severity_poses'] 
                            and p[1] >= self.SUSPICIOUS_VALIDATION['suspicious_confidence_threshold']]
                
                if high_severity:
                    track['suspicious_frame_count'] += 1
                    
                    if track['suspicious_frame_count'] >= 3:
                        transition_to_suspicious = True
                        transition_reason = f"HIGH_SEVERITY_POSE: {[p[2] for p in high_severity]}"
            else:
                if not zone_penetration:
                    track['suspicious_frame_count'] = max(0, track['suspicious_frame_count'] - 1)
                    track['last_normal_frame'] = current_frame
            
            # CEK 3: KONSISTENSI DALAM WINDOW
            if not transition_to_suspicious and len(track['suspicious_buffer']) >= self.SUSPICIOUS_VALIDATION['pose_consistency_window']:
                recent_suspicious = [b for b in track['suspicious_buffer'] if b['has_suspicious']]
                suspicious_ratio = len(recent_suspicious) / len(track['suspicious_buffer'])
                
                all_poses = []
                for b in recent_suspicious:
                    all_poses.extend([p[0] for p in b['poses']])
                unique_poses = len(set(all_poses))
                
                if suspicious_ratio >= 0.50 and unique_poses >= self.SUSPICIOUS_VALIDATION['min_unique_poses']:
                    avg_confidence = np.mean([p[1] for b in recent_suspicious for p in b['poses']]) if recent_suspicious else 0
                    
                    if avg_confidence >= self.SUSPICIOUS_VALIDATION['suspicious_confidence_threshold']:
                        transition_to_suspicious = True
                        transition_reason = f"CONSISTENT: ratio={suspicious_ratio:.1%}, poses={unique_poses}"
            
            # EXECUTE TRANSITION
            if transition_to_suspicious:
                track['phase'] = DetectionPhase.SUSPICIOUS_MOVEMENT
                track['phase_start_frame'] = current_frame
                
                if self.debug_mode:
                    print(f"🔴 Track {track_id}: GRABBING -> SUSPICIOUS")
                    print(f"    Reason: {transition_reason}")
                    if zone_details:
                        for z in zone_details:
                            print(f"    - {z['zone']}: {z['hand']} (depth: {z['depth']:.1%}, frames: {z['frames_in_zone']})")
            
            # CEK 4: TIMEOUT - NORMAL BEHAVIOR
            frames_since_grab = current_frame - track['grab_frame']
            
            if frames_since_grab >= self.SUSPICIOUS_VALIDATION['timeout_normal_behavior']:
                normal_frames = len([b for b in track['suspicious_buffer'] if not b['has_suspicious']])
                
                if normal_frames >= len(track['suspicious_buffer']) * 0.7:
                    track['phase'] = DetectionPhase.IDLE
                    self.reset_track(track_id)
                    
                    if self.debug_mode:
                        print(f"⚪ Track {track_id}: GRABBING -> IDLE (Normal behavior)")
                    return False, [], []
        
        # FASE 4: SUSPICIOUS_MOVEMENT
        elif current_phase == DetectionPhase.SUSPICIOUS_MOVEMENT:
            suspicious_poses = self.detect_suspicious_poses(keypoints)
            zone_penetration, zone_details, _, zone_poses = self.detect_zone_penetration(track_id, keypoints)
            
            # Combine all poses
            all_poses = suspicious_poses + zone_poses
            
            if all_poses or zone_penetration:
                self.update_suspicion_score(track_id, all_poses, zone_details, current_frame)
                
                should_alert, reasons = self.should_alert(track_id, all_poses, current_frame)
                
                if should_alert:
                    track['phase'] = DetectionPhase.ALERT
                    return True, all_poses, reasons
            else:
                if current_frame - track['phase_start_frame'] > 10:
                    track['phase'] = DetectionPhase.IDLE
                    self.reset_track(track_id)
        
        # FASE 5: ALERT
        elif current_phase == DetectionPhase.ALERT:
            if current_frame - track['last_alert_frame'] > self.SUSPICIOUS_THRESHOLDS['alert_cooldown'] * 3:
                track['phase'] = DetectionPhase.IDLE
                self.reset_track(track_id)
        
        return False, [], []
    
    def reset_track(self, track_id):
        """Reset tracking untuk person tertentu"""
        track = self.person_tracks[track_id]
        track['hand_extended'] = False
        track['hand_extended_frames'] = 0
        track['grab_detected'] = False
        track['grabbed_hand'] = None
        track['suspicion_score'] = 0
        track['consecutive_suspicious'] = 0
        track['suspicious_poses'].clear()
        track['wrist_positions'].clear()
        track['wrist_in_zone_frames'].clear()
        track['max_zone_depth'].clear()
        track['zone_penetration_detected'] = False
        track['zone_penetration_frames'] = 0
        track['zone_penetration_zones'].clear()
        track['zone_consecutive_frames'] = 0  
        track['suspicious_frame_count'] = 0
    
    def update_suspicion_score(self, track_id, suspicious_poses, zone_details, current_frame):
        """Update suspicion score dengan zone detection"""
        track = self.person_tracks[track_id]
        
        track['suspicion_score'] = max(0, track['suspicion_score'] - self.SUSPICIOUS_THRESHOLDS['score_decay'])
        track['total_frames_tracked'] += 1
        
        # Zone detection scoring
        if zone_details:
            for zone_detail in zone_details:
                zone_name = zone_detail['zone']
                depth = zone_detail['depth']
                confidence = zone_detail['confidence']
                
                if 'pants' in zone_name:
                    base_score = 30  
                    depth_multiplier = 1.5
                else:
                    base_score = 26  
                    depth_multiplier = 1.3
                
                zone_score = base_score * confidence * (1 + depth * depth_multiplier)
                track['suspicion_score'] += zone_score
                
                if self.debug_mode:
                    print(f"  Zone {zone_name}: +{zone_score:.1f} (depth: {depth:.1%}, conf: {confidence:.2f})")
        
        if suspicious_poses:
            track['suspicious_poses'].append({
                'frame': current_frame,
                'poses': suspicious_poses
            })
            
            track['consecutive_suspicious'] += 1
            
            for pose_type, confidence, _ in suspicious_poses:
                track['pose_counts'][pose_type] += 1
                
                if pose_type == SuspiciousPose.HIDING_UNDER_CLOTHING:
                    track['suspicion_score'] += 25 * confidence
                elif pose_type == SuspiciousPose.PUTTING_IN_PANTS_POCKET:
                    track['suspicion_score'] += 22 * confidence
                elif pose_type == SuspiciousPose.SQUATTING_LOW:
                    track['suspicion_score'] += 18 * confidence
                elif pose_type == SuspiciousPose.BENDING_DOWN:
                    track['suspicion_score'] += 15 * confidence
                elif pose_type == SuspiciousPose.REACHING_WAIST_BACK:
                    track['suspicion_score'] += 16 * confidence
                elif pose_type == SuspiciousPose.CONCEALING_AT_WAIST:
                    track['suspicion_score'] += 14 * confidence
                elif pose_type == SuspiciousPose.CROUCHING:
                    track['suspicion_score'] += 12 * confidence
                elif pose_type in [SuspiciousPose.ZONE_PANTS_POCKET_LEFT, SuspiciousPose.ZONE_PANTS_POCKET_RIGHT]:
                    track['suspicion_score'] += 28 * confidence
                elif pose_type in [SuspiciousPose.ZONE_JACKET_POCKET_LEFT, SuspiciousPose.ZONE_JACKET_POCKET_RIGHT]:
                    track['suspicion_score'] += 24 * confidence
                
            if track['consecutive_suspicious'] >= self.SUSPICIOUS_THRESHOLDS['min_consecutive_for_bonus']:
                track['suspicion_score'] += self.SUSPICIOUS_THRESHOLDS['continuous_pose_bonus']
            
            unique_poses = len(set(p[0] for p in suspicious_poses))
            if unique_poses >= 2:
                track['suspicion_score'] += 20
        else:
            track['consecutive_suspicious'] = 0
        
        track['suspicion_score'] = min(track['suspicion_score'], 100)
        
        suspicious_frame_count = sum(1 for p in track['suspicious_poses'])
        track['suspicious_ratio'] = suspicious_frame_count / max(track['total_frames_tracked'], 1)
        
        return track['suspicion_score']
    
    def should_alert(self, track_id, suspicious_poses, current_frame):
        """Tentukan apakah harus trigger alert"""
        track = self.person_tracks[track_id]
        
        if not track['grab_detected']:
            return False, []
        
        frames_since_alert = current_frame - track['last_alert_frame']
        if frames_since_alert < self.SUSPICIOUS_THRESHOLDS['alert_cooldown']:
            return False, []
        
        if track['total_frames_tracked'] < self.SUSPICIOUS_THRESHOLDS['min_tracking_frames']:
            return False, []
        
        reasons = []
        
        high_conf_poses = [p for p in suspicious_poses 
                          if p[1] >= self.SUSPICIOUS_THRESHOLDS['high_confidence_threshold']]
        if high_conf_poses and track['consecutive_suspicious'] >= 6:
            reasons = [f"GRAB+{p[2]}" for p in high_conf_poses[:2]]
            return True, reasons

        if track['zone_penetration_detected']:
            zone_pose_counts = sum(1 for pose_type in track['pose_counts'] 
                                if pose_type in [SuspiciousPose.ZONE_PANTS_POCKET_LEFT,
                                                SuspiciousPose.ZONE_PANTS_POCKET_RIGHT,
                                                SuspiciousPose.ZONE_JACKET_POCKET_LEFT,
                                                SuspiciousPose.ZONE_JACKET_POCKET_RIGHT])
            
            if zone_pose_counts >= 8: 
                for zone_name in track['zone_penetration_zones']:
                    reasons.append(f"GRAB + {zone_name.replace('_', ' ').upper()}")
                return True, reasons
        
        if (track['suspicion_score'] >= self.SUSPICIOUS_THRESHOLDS['score_threshold'] and 
            track['suspicious_ratio'] >= self.SUSPICIOUS_THRESHOLDS['suspicious_ratio_threshold']):
            
            top_poses = sorted(track['pose_counts'].items(), 
                             key=lambda x: x[1], reverse=True)[:2]
            for pose_type, count in top_poses:
                if count > 0:
                    reasons.append(f"{pose_type.value}: {count}x")
            
            recent_count = sum(1 for p in track['suspicious_poses'] 
                             if p['frame'] > current_frame - 30)
            if recent_count >= self.SUSPICIOUS_THRESHOLDS['suspicious_frame_count']:
                reasons.append(f"GRABBED + {recent_count}f suspicious")
                return True, reasons
        
        if (len(suspicious_poses) >= 2 and 
            track['suspicion_score'] > 55 and
            track['consecutive_suspicious'] >= 8):
            reasons = [f"GRAB+{p[2][:25]}" for p in suspicious_poses[:2]]
            return True, reasons
        
        return False, []
    
    def save_alert_clip(self, track_id, alert_info, current_frame):
        """Save video clip: 5s sebelum + 5s setelah"""
        try:
            clips_dir = "alert_clips"
            os.makedirs(clips_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"shoplifting_track{track_id}_{timestamp}"
            
            frames_before = int(5 * self.fps)
            frames_after = int(5 * self.fps)
            
            frames_pre_alert = []
            if len(self.frame_buffer) >= frames_before:
                frames_pre_alert = list(self.frame_buffer)[-frames_before:]
            else:
                frames_pre_alert = list(self.frame_buffer)
            
            print(f"📹 Recording alert clip for Track {track_id}...")
            print(f"   - Pre-alert frames: {len(frames_pre_alert)}")
            print(f"   - Waiting for {frames_after} post-alert frames...")
            
            self.recording_alerts[track_id] = {
                'start_frame': current_frame,
                'frames_before': frames_pre_alert,
                'frames_after': [],
                'frames_needed': frames_after,
                'alert_info': alert_info,
                'base_filename': base_filename
            }
            
            return base_filename
            
        except Exception as e:
            print(f"❌ Error preparing alert clip: {e}")
            return None
    
    def finalize_alert_clip(self, track_id):
        """Finalize dan save clip"""
        if track_id not in self.recording_alerts:
            return None
        
        try:
            recording = self.recording_alerts[track_id]
            base_filename = recording['base_filename']
            
            all_frames = recording['frames_before'] + recording['frames_after']
            
            if len(all_frames) < 30:
                print(f"⚠️ Not enough frames for Track {track_id}")
                del self.recording_alerts[track_id]
                return None
            
            clips_dir = "alert_clips"
            
            video_filename = os.path.join(clips_dir, f"{base_filename}.mp4")
            
            first_frame = all_frames[0]
            height, width = first_frame.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_filename, fourcc, self.fps, (width, height))
            
            if not out.isOpened():
                print(f"❌ Cannot open video writer for Track {track_id}")
                del self.recording_alerts[track_id]
                return None
            
            for frame in all_frames:
                out.write(frame)
            
            out.release()
            
            json_filename = os.path.join(clips_dir, f"{base_filename}.json")
            
            alert_info = recording['alert_info']
            pose_descriptions = self._generate_pose_descriptions(alert_info)
            
            clip_info = {
                'alert_info': alert_info,
                'clip_info': {
                    'video_file': f"{base_filename}.mp4",
                    'total_frames': len(all_frames),
                    'pre_alert_frames': len(recording['frames_before']),
                    'post_alert_frames': len(recording['frames_after']),
                    'duration_seconds': len(all_frames) / self.fps,
                    'fps': self.fps,
                    'resolution': f"{width}x{height}",
                    'created_at': datetime.now().isoformat()
                },
                'detection_summary': {
                    'track_id': track_id,
                    'phase_sequence': alert_info.get('phase', 'unknown'),
                    'grab_frame': alert_info.get('grab_frame', 0),
                    'alert_frame': alert_info.get('frame', 0),
                    'grabbed_hand': alert_info.get('grabbed_hand', 'unknown'),
                    'suspicion_score': alert_info.get('suspicion_score', 0),
                    'suspicious_ratio': alert_info.get('suspicious_ratio', 0),
                    'reasons': alert_info.get('reasons', []),
                    'pose_counts': alert_info.get('pose_counts', {}),
                    'zone_penetration': alert_info.get('zone_penetration_detected', False),  # BARU
                    'zone_penetration_zones': alert_info.get('zone_penetration_zones', [])
                },
                'behavior_analysis': {
                    'description': pose_descriptions['full_description'],
                    'action_sequence': pose_descriptions['action_sequence'],
                    'suspicious_actions': pose_descriptions['suspicious_actions'],
                    'dominant_pose': pose_descriptions['dominant_pose'],
                    'severity_level': pose_descriptions['severity_level'],
                    'detailed_breakdown': pose_descriptions['detailed_breakdown']
                }
            }
            
            with open(json_filename, 'w') as f:
                json.dump(clip_info, f, indent=2)
            
            if os.path.exists(video_filename) and os.path.exists(json_filename):
                self.alert_clips_saved.append(base_filename)
                print(f"✅ Alert clip saved:")
                print(f"   📹 Video: {video_filename}")
                print(f"   📄 JSON:  {json_filename}")
                print(f"   ⏱️  Duration: {len(all_frames) / self.fps:.1f}s ({len(all_frames)} frames)")
                print(f"   📝 Behavior: {pose_descriptions['full_description']}")
                
                del self.recording_alerts[track_id]
                return base_filename
            else:
                print(f"❌ Failed to save files for Track {track_id}")
                del self.recording_alerts[track_id]
                return None
            
        except Exception as e:
            print(f"❌ Error finalizing clip for Track {track_id}: {e}")
            import traceback
            traceback.print_exc()
            if track_id in self.recording_alerts:
                del self.recording_alerts[track_id]
            return None
    
    def _generate_pose_descriptions(self, alert_info):
        """Generate deskripsi lengkap tentang pose/behavior"""
        pose_counts = alert_info.get('pose_counts', {})
        reasons = alert_info.get('reasons', [])
        grabbed_hand = alert_info.get('grabbed_hand', 'unknown')
        suspicion_score = alert_info.get('suspicion_score', 0)
        zone_penetration = alert_info.get('zone_penetration_detected', False)
        zone_names = alert_info.get('zone_penetration_zones', [])
        
        pose_descriptions = {
            'bending_down': 'membungkuk ke bawah',
            'crouching': 'berjongkok',
            'hiding_under_clothing': 'memasukkan sesuatu ke dalam pakaian',
            'concealing_at_waist': 'menyembunyikan sesuatu di area pinggang',
            'reaching_pocket': 'meraih kantong',
            'hands_near_body': 'tangan dekat dengan tubuh',
            'putting_in_pants_pocket': 'memasukkan sesuatu ke kantong celana',
            'hands_behind_back': 'meletakkan tangan di belakang punggung',
            'squatting_low': 'jongkok rendah',
            'reaching_waist_back': 'meraih area pinggang belakang',
            'zone_pants_pocket_left': 'memasukkan tangan ke kantong celana kiri',
            'zone_pants_pocket_right': 'memasukkan tangan ke kantong celana kanan',
            'zone_jacket_pocket_left': 'memasukkan tangan ke kantong jaket kiri',
            'zone_jacket_pocket_right': 'memasukkan tangan ke kantong jaket kanan'
        }
        
        sorted_poses = sorted(pose_counts.items(), key=lambda x: x[1], reverse=True)
        
        action_sequence = []
        action_sequence.append(f"1. Mengangkat tangan {grabbed_hand} untuk mengambil barang")

        if zone_penetration and zone_names:
            for zone in zone_names:
                zone_desc = zone.replace('_', ' ').title()
                action_sequence.append(f"2. 🚨 ZONE DETECTION: Tangan masuk ke {zone_desc}")

        if zone_penetration:
            zone_desc_list = [z.replace('_', ' ') for z in zone_names]
            full_description = (
                f"🚨 ZONE PENETRATION DETECTED! Orang terdeteksi mengambil barang dengan tangan {grabbed_hand}, "
                f"kemudian langsung memasukkan tangan ke zona: {', '.join(zone_desc_list)}. "
            )
            
        for i, (pose_key, count) in enumerate(sorted_poses[:3], start=2):
            pose_name = str(pose_key).replace('SuspiciousPose.', '').lower() if hasattr(pose_key, '__class__') else str(pose_key).lower()
            desc = pose_descriptions.get(pose_name, pose_name.replace('_', ' '))
            action_sequence.append(f"{i}. Terdeteksi {desc} sebanyak {count} kali")
        
        suspicious_actions = []
        for pose_key, count in sorted_poses:
            pose_name = str(pose_key).replace('SuspiciousPose.', '').lower() if hasattr(pose_key, '__class__') else str(pose_key).lower()
            desc = pose_descriptions.get(pose_name, pose_name.replace('_', ' '))
            suspicious_actions.append({
                'action': desc,
                'count': count,
                'pose_type': pose_name
            })
        
        if sorted_poses:
            dominant_pose_key = sorted_poses[0][0]
            dominant_pose_key_str = str(dominant_pose_key).replace('SuspiciousPose.', '').lower() if hasattr(dominant_pose_key, '__class__') else str(dominant_pose_key).lower()
            dominant_pose = pose_descriptions.get(dominant_pose_key_str, dominant_pose_key_str.replace('_', ' '))
            dominant_count = sorted_poses[0][1]
        else:
            dominant_pose = "tidak teridentifikasi"
            dominant_count = 0
        
        if sorted_poses:
            top_3_poses = []
            for pose_key, _ in sorted_poses[:3]:
                pose_name = str(pose_key).replace('SuspiciousPose.', '').lower() if hasattr(pose_key, '__class__') else str(pose_key).lower()
                top_3_poses.append(pose_descriptions.get(pose_name, pose_name.replace('_', ' ')))
            
            full_description = (
                f"Orang terdeteksi mengambil barang dengan tangan {grabbed_hand}, "
                f"kemudian melakukan gerakan mencurigakan: {', '.join(top_3_poses)}. "
                f"Pose dominan adalah '{dominant_pose}' yang terdeteksi {dominant_count} kali."
            )
        else:
            full_description = (
                f"Orang terdeteksi mengambil barang dengan tangan {grabbed_hand} "
                f"dan melakukan gerakan mencurigakan setelahnya."
            )
        
        if suspicion_score >= 85:
            severity = "SANGAT TINGGI - Kemungkinan besar shoplifting"
        elif suspicion_score >= 75:
            severity = "TINGGI - Perilaku sangat mencurigakan"
        elif suspicion_score >= 65:
            severity = "SEDANG - Perilaku cukup mencurigakan"
        else:
            severity = "RENDAH - Perilaku agak mencurigakan"
        
        detailed_breakdown = {
            'initial_action': f"Mengangkat tangan {grabbed_hand} untuk mengambil barang dari rak/shelf",
            'grabbing_confirmed': True,
            'suspicious_movements': [],
            'concealment_method': None,
            'body_position': [],
            'zone_penetration': zone_penetration,  #
            'zone_details': zone_names if zone_penetration else []
        }
        
        for pose_key, count in sorted_poses:
            pose_name = str(pose_key).replace('SuspiciousPose.', '').lower() if hasattr(pose_key, '__class__') else str(pose_key).lower()
            desc = pose_descriptions.get(pose_name, pose_name)
            
            if 'hiding' in pose_name or 'pocket' in pose_name or 'concealing' in pose_name:
                if not detailed_breakdown['concealment_method']:
                    detailed_breakdown['concealment_method'] = desc
            
            if 'bending' in pose_name or 'crouch' in pose_name or 'squat' in pose_name:
                detailed_breakdown['body_position'].append(desc)
            
            detailed_breakdown['suspicious_movements'].append({
                'movement': desc,
                'frequency': count,
                'severity': 'high' if count > 10 else ('medium' if count > 5 else 'low')
            })
        
        if not detailed_breakdown['concealment_method']:
            detailed_breakdown['concealment_method'] = "Metode penyembunyian tidak teridentifikasi dengan jelas"
        
        if not detailed_breakdown['body_position']:
            detailed_breakdown['body_position'] = ["Posisi tubuh normal/berdiri"]
        
        return {
            'full_description': full_description,
            'action_sequence': action_sequence,
            'suspicious_actions': suspicious_actions,
            'dominant_pose': {
                'pose': dominant_pose,
                'count': dominant_count
            },
            'severity_level': severity,
            'detailed_breakdown': detailed_breakdown
        }
    
    def update_recording_alerts(self, frame):
        """Update semua recording alerts dengan frame baru"""
        to_finalize = []
        
        for track_id, recording in list(self.recording_alerts.items()):
            frames_after = recording['frames_after']
            frames_needed = recording['frames_needed']
            
            if len(frames_after) < frames_needed:
                recording['frames_after'].append(frame.copy())
                
                if len(frames_after) >= frames_needed:
                    to_finalize.append(track_id)
        
        for track_id in to_finalize:
            self.finalize_alert_clip(track_id)
    
    def process_frame(self, frame):
        """Process frame dengan zone visualization"""
        self.frame_count += 1
        self.frame_buffer.append(frame.copy())
        self.update_recording_alerts(frame)
        
        results = self.pose_model.track(
            frame,
            persist=True,
            verbose=False,
            imgsz=640
        )
        
        alert_persons = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints_data = results[0].keypoints.data.cpu().numpy()
            
            for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x1, y1, x2, y2 = map(int, box)
                keypoints = keypoints_data[i]
                
                track = self.person_tracks[track_id]
                track['current_keypoints'] = keypoints
                
                if not track['pocket_zones']:
                    self.initialize_pocket_zones(track_id, keypoints)
                
                is_alert, suspicious_poses, reasons = self.update_phase(
                    track_id, keypoints, self.frame_count
                )
                
                # Draw keypoints
                for kp in keypoints:
                    if kp[2] > 0.6:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                
                # Draw skeleton
                connections = [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 11), (6, 12), (11, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16)
                ]
                for start_idx, end_idx in connections:
                    if keypoints[start_idx][2] > 0.6 and keypoints[end_idx][2] > 0.6:
                        start = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                        end = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                        cv2.line(frame, start, end, (0, 255, 0), 2)
                
                # DRAW POCKET ZONES (jika grab detected)
                if track['grab_detected'] and track['pocket_zones']:
                    self.update_pocket_zones(track_id, keypoints)
                    
                    for zone_name, zone in track['pocket_zones'].items():
                        if zone.zone_box:
                            x1z, y1z, x2z, y2z = zone.zone_box
                            
                            if track['wrist_in_zone_frames'][zone_name] > 0:
                                color = (0, 0, 255)
                                thickness = 3
                                alpha = 0.4
                            else:
                                color = (255, 165, 0)
                                thickness = 2
                                alpha = 0.2
                            
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (x1z, y1z), (x2z, y2z), color, -1)
                            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                            
                            cv2.rectangle(frame, (x1z, y1z), (x2z, y2z), color, thickness)
                            
                            label = zone_name.replace('_', ' ').upper()
                            frames_in = track['wrist_in_zone_frames'][zone_name]
                            if frames_in > 0:
                                label += f" ({frames_in}f)"
                            
                            cv2.putText(frame, label, (x1z, y1z - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # ALERT TRIGGERED
                if is_alert and not track['alert_triggered']:
                    alert_info = {
                        'timestamp': datetime.now().isoformat(),
                        'frame': self.frame_count,
                        'track_id': track_id,
                        'phase': track['phase'].value,
                        'grab_frame': track['grab_frame'],
                        'grabbed_hand': track['grabbed_hand'],
                        'suspicion_score': track['suspicion_score'],
                        'suspicious_ratio': track['suspicious_ratio'],
                        'total_frames': track['total_frames_tracked'],
                        'reasons': reasons,
                        'zone_penetration_detected': track['zone_penetration_detected'],  # BARU
                        'zone_penetration_zones': track['zone_penetration_zones'],
                        'pose_counts': dict(track['pose_counts'])
                    }
                    
                    self.alert_log.append(alert_info)
                    track['alert_triggered'] = True
                    track['last_alert_frame'] = self.frame_count
                    
                    alert_persons.append(track_id)
                    
                    clip_name = self.save_alert_clip(track_id, alert_info, self.frame_count)
                    
                    if self.debug_mode:
                        print(f"\n🚨 SHOPLIFTING ALERT: Track {track_id}")
                        print(f"   Grabbed: Frame {track['grab_frame']} ({track['grabbed_hand']} hand)")
                        print(f"   Score: {track['suspicion_score']:.1f}")
                        print(f"   Reasons: {reasons}")
                        if clip_name:
                            print(f"   Clip: {clip_name}")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    label = f"SHOPLIFTING! ID:{track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    
                    y_offset = y2 + 25
                    for reason in reasons[:2]:
                        cv2.putText(frame, reason, (x1, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        y_offset += 20
                
                else:
                    phase = track['phase']
                    score = track['suspicion_score']
                    
                    if phase == DetectionPhase.IDLE:
                        color = (0, 255, 0)
                        label = f"ID:{track_id} [IDLE]"
                        thickness = 2
                    elif phase == DetectionPhase.REACHING_SHELF:
                        color = (0, 255, 255)
                        label = f"ID:{track_id} [REACHING {track['grabbed_hand']}]"
                        thickness = 3
                    elif phase == DetectionPhase.GRABBING:
                        color = (0, 165, 255)
                        label = f"ID:{track_id} [GRABBED!]"
                        thickness = 3
                    elif phase == DetectionPhase.SUSPICIOUS_MOVEMENT:
                        color = (0, 0, 255)
                        label = f"ID:{track_id} [SUSPICIOUS {score:.0f}]"
                        thickness = 4
                    elif phase == DetectionPhase.ALERT:
                        color = (0, 0, 255)
                        label = f"ID:{track_id} [ALERTED]"
                        thickness = 4
                    else:
                        color = (0, 255, 0)
                        label = f"ID:{track_id}"
                        thickness = 2
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    if phase == DetectionPhase.SUSPICIOUS_MOVEMENT:
                        info_text = f"Score:{score:.0f} Ratio:{track['suspicious_ratio']:.1%}"
                        cv2.putText(frame, info_text, (x1, y2 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    if track['grab_detected'] and phase != DetectionPhase.IDLE:
                        frames_since_grab = self.frame_count - track['grab_frame']
                        grab_info = f"Grabbed {frames_since_grab}f ago ({track['grabbed_hand']})"
                        cv2.putText(frame, grab_info, (x1, y2 + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Alert banner
        if alert_persons:
            text = f"SHOPLIFTING DETECTED - {len(alert_persons)} PERSON(S)"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            
            cv2.rectangle(frame, (text_x - 20, 10), 
                         (text_x + text_size[0] + 20, 60), (0, 0, 255), -1)
            cv2.putText(frame, text, (text_x, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Show recording status
        if self.recording_alerts:
            recording_text = f"RECORDING: {len(self.recording_alerts)} clip(s)"
            cv2.putText(frame, recording_text, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame, alert_persons
    
    def save_session_log(self):
        """Save detection log"""
        if self.alert_log:
            log_data = {
                'session_info': {
                    'start_time': self.session_start.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_frames': self.frame_count,
                    'total_alerts': len(self.alert_log),
                    'method': 'POSE_WITH_GRABBING_DETECTION_v6_PLUS_ZONES',
                    'grab_thresholds': self.GRAB_THRESHOLDS,
                    'suspicious_thresholds': self.SUSPICIOUS_THRESHOLDS,
                    'zone_thresholds': self.ZONE_THRESHOLDS
                },
                'alerts': self.alert_log
            }
            
            filename = f"shoplifting_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"\n✅ Log saved: {filename}")
            return filename
        return None


def main():
    print("=" * 80)
    print("SHOPLIFTING DETECTION WITH GRABBING PHASE + POCKET ZONES v6+")
    print("5-PHASE DETECTION: REACHING -> GRABBING -> ZONE PENETRATION -> SUSPICIOUS -> ALERT")
    print("=" * 80)
    print("\n📋 DETECTION PHASES:")
    print("  1. IDLE           - Normal behavior")
    print("  2. REACHING SHELF - Tangan meraih")
    print("  3. GRABBING       - Hand closing/menggenggam")
    print("  4. SUSPICIOUS     - Gerakan mencurigakan setelah grab:")
    print("                      • Zone penetration (kantong celana/jaket)")
    print("                      • Memasukkan ke baju")
    print("                      • Membungkuk/jongkok")
    print("                      • Pose mencurigakan lain")
    print("  5. ALERT          - Alert triggered!")
    print("\n NEW FEATURES:")
    print("  • POCKET ZONE DETECTION - Dynamic zones for pants & jacket pockets")
    print("  • Multi-level penetration tracking - Depth measurement (0-1)")
    print("  • Automatic zone initialization - Based on body keypoints")
    print("  • Zone-based alert - Instant suspicious on high depth penetration")
    print("  • Visual zone overlay - Real-time zone visualization")
    print("\n ZONE CONFIGURATION:")
    print("  • Pants pockets (left/right) - Bottom area, 30% body width")
    print("  • Jacket pockets (left/right) - Mid-body, 25% body width")
    print("  • Dynamic sizing - Adapts to person height/width")
    print("  • Penetration depth - 0.3+ triggers medium, 0.6+ triggers high severity")
    print("=" * 80)
    
    debug = input("\nDebug mode? (y/n) [n]: ").lower() == 'y'
    
    try:
        detector = ShopliftingPoseDetectorWithGrab(
            pose_model="yolo11m-pose.pt",
            debug_mode=debug
        )
    except Exception as e:
        print(f"\nInitialization failed: {e}")
        return
    
    print("\nVideo source:")
    print("1. Webcam")
    print("2. Video file")
    print("3. RTSP/IP Camera")
    choice = input("Choose (1-3) [1]: ").strip() or "1"
    
    if choice == "2":
        path = input("Video path: ").strip()
        if not os.path.exists(path):
            print("File not found")
            return
        cap = cv2.VideoCapture(path)
        source = f"Video: {os.path.basename(path)}"
    elif choice == "3":
        rtsp_url = input("RTSP URL: ").strip()
        if not rtsp_url:
            print("RTSP URL required")
            return
        cap = cv2.VideoCapture(rtsp_url)
        source = f"RTSP: {rtsp_url}"
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        source = "Webcam"
    
    if not cap.isOpened():
        print(f"Cannot open {source}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        detector.fps = fps
    
    print(f"✅ {source} ready | FPS: {detector.fps}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save log")
    print("  'd' - Toggle debug overlay")
    print("  'r' - Reset all tracks")
    print("=" * 80)
    
    total_alerts = 0
    start_time = time.time()
    frame_times = deque(maxlen=30)
    show_debug_info = True
    
    try:
        while True:
            t_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                if choice == "3":
                    print("Connection lost, reconnecting...")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(rtsp_url)
                    continue
                else:
                    break
            
            frame = cv2.resize(frame, (1280, 720))
            
            processed, alerts = detector.process_frame(frame)
            
            if alerts:
                total_alerts += len(alerts)
            
            frame_times.append(time.time() - t_start)
            current_fps = 1.0 / np.mean(frame_times) if frame_times else 0
            
            cv2.putText(processed, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if total_alerts > 0:
                cv2.putText(processed, f"TOTAL ALERTS: {total_alerts}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if show_debug_info:
                y_pos = 100
                overlay = processed.copy()
                
                cv2.rectangle(overlay, (5, y_pos - 20), (450, y_pos + 200), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, processed, 0.4, 0, processed)
                
                cv2.putText(processed, "=== PHASE STATUS ===", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 30
                
                active_tracks = [(tid, t) for tid, t in detector.person_tracks.items() 
                               if t['total_frames_tracked'] > 0 or t['phase'] != DetectionPhase.IDLE]
                
                if not active_tracks:
                    cv2.putText(processed, "No active tracks", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    for track_id, track_data in active_tracks[:5]:
                        phase_color = {
                            DetectionPhase.IDLE: (0, 255, 0),
                            DetectionPhase.REACHING_SHELF: (0, 255, 255),
                            DetectionPhase.GRABBING: (0, 165, 255),
                            DetectionPhase.SUSPICIOUS_MOVEMENT: (0, 0, 255),
                            DetectionPhase.ALERT: (0, 0, 255)
                        }.get(track_data['phase'], (255, 255, 255))
                        
                        phase_text = f"ID{track_id}: {track_data['phase'].value.upper()}"
                        cv2.putText(processed, phase_text, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 1)
                        y_pos += 20
                        
                        if track_data['phase'] == DetectionPhase.REACHING_SHELF:
                            info = f"  Hand: {track_data['grabbed_hand']} ({track_data['hand_extended_frames']}f)"
                            cv2.putText(processed, info, (10, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, phase_color, 1)
                            y_pos += 18
                        elif track_data['grab_detected']:
                            frames_since_grab = detector.frame_count - track_data['grab_frame']
                            info = f"  Grabbed: {frames_since_grab}f ago | Score: {track_data['suspicion_score']:.0f}"
                            cv2.putText(processed, info, (10, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, phase_color, 1)
                            y_pos += 18
            
            cv2.imshow("Shoplifting Detection", processed)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                detector.save_session_log()
            elif key == ord('d'):
                show_debug_info = not show_debug_info
                print(f"Debug overlay: {'ON' if show_debug_info else 'OFF'}")
            elif key == ord('r'):
                detector.person_tracks.clear()
                print("All tracks reset")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if detector.recording_alerts:
            print(f"\nFinalizing {len(detector.recording_alerts)} pending clip(s)...")
            for track_id in list(detector.recording_alerts.keys()):
                detector.finalize_alert_clip(track_id)
        
        cap.release()
        cv2.destroyAllWindows()
        
        runtime = time.time() - start_time
        print("\n" + "=" * 80)
        print("SESSION SUMMARY")
        print("=" * 80)
        print(f"Runtime: {runtime:.1f}s")
        print(f"Frames Processed: {detector.frame_count}")
        print(f"Total Alerts: {total_alerts}")
        print(f"Clips Saved: {len(detector.alert_clips_saved)}")
        print(f"Average FPS: {detector.frame_count / runtime:.1f}")
        if runtime > 0:
            print(f"Alert Rate: {(total_alerts / (runtime / 60)):.2f} alerts/minute")
        
        if detector.alert_clips_saved:
            print("\nSaved Alert Clips:")
            for i, clip in enumerate(detector.alert_clips_saved, 1):
                print(f"   {i}. {clip}.mp4 + {clip}.json")
        
        print("=" * 80)
        
        detector.save_session_log()
        print("\nSession completed successfully")


if __name__ == "__main__":
    main()