import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from collections import defaultdict, deque
import time
import math
import os
import json
import mediapipe as mp


class PoseShopliftingDetector:
    def __init__(self, model_path="yolo11s.pt", pose_model="yolo11n-pose.pt"):
        print("🚀 Initializing Improved Pose-Based Shoplifting Detector v3.1...")

        self.detection_model = YOLO(model_path)
        self.pose_model = YOLO(pose_model)
        print("✅ YOLO detection and pose models loaded")
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.3    
        )
        print("✅ MediaPipe hands model loaded")
        
        self.frame_count = 0
        self.zones = {}
        self.zones_calibrated = False
        
        self.track_history = defaultdict(lambda: {
            'positions': deque(maxlen=50),
            'timestamps': deque(maxlen=50),
            'pose_keypoints': deque(maxlen=30),
            'hand_positions': deque(maxlen=20),
            'suspicious_actions': [],
            'action_scores': {
                'reaching_to_pocket': 0,
                'hiding_in_bag': 0,
                'concealing_object': 0,
                'looking_around': 0,
                'rapid_movements': 0,
                'hand_near_products': 0,
                'suspicious_body_language': 0
            },
            'pose_analysis': {
                'shoulder_angle_history': deque(maxlen=10),
                'elbow_angle_history': deque(maxlen=10),
                'hand_movement_speed': deque(maxlen=10),
                'body_orientation': deque(maxlen=10),
                'head_direction': deque(maxlen=10),
                'crouching_detected': False,
                'hands_near_torso_time': 0,
                'rapid_hand_movements': 0,
                'pocket_interaction_frames': 0,
                'bag_interaction_frames': 0
            },
            'object_interaction': {
                'holding_objects': False,
                'object_disappeared': False,
                'hand_object_correlation': deque(maxlen=15),
                'pickup_detected': False,
                'concealment_detected': False
            },
            'behavioral_flags': {
                'looking_around_suspiciously': False,
                'avoiding_cameras': False,
                'nervous_behavior': False,
                'group_coordination': False
            },
            'suspicious_score': 0,
            'alert_level': 'LOW',
            'last_action_time': time.time(),
            'total_time_tracked': 0
        })
        
        self.pose_thresholds = {
            'pocket_reach_angle': 65,          
            'bag_concealment_duration': 20,     
            'rapid_movement_threshold': 80,    
            'hand_near_torso_ratio': 1.0,     
            'suspicious_score_threshold': 120,  
            'head_turn_frequency': 12,          
            'crouch_duration': 4,              
            'object_disappearance_threshold': 10, 
            'normal_shopping_time': 30,        
            'concealment_confirmation_time': 6, 
            'multiple_action_threshold': 7,    
            'pocket_interaction_threshold': 75, 
            'bag_interaction_threshold': 75     
        }

        self.action_weights = {
            'concealing_object': 1.0,  
            'hiding_in_bag': 1.0,      
            'rapid_movement': 1.0,     
            'pocket_reach': 1.0,       
            'loitering': 1.0,         
            'looking_around': 1.0      
        }

    
        self.keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
    
    def auto_calibrate_zones(self, frame_width, frame_height):
        """Auto-calculate zones based on frame dimensions"""
        print(f"🎯 Auto-calibrating zones for {frame_width}x{frame_height}")
        
        self.zones = {
            'entrance': (0, 0, int(frame_width * 0.2), frame_height),
            'products_high': (int(frame_width * 0.2), 0, int(frame_width * 0.8), int(frame_height * 0.4)),
            'products_low': (int(frame_width * 0.2), int(frame_height * 0.4), int(frame_width * 0.8), int(frame_height * 0.8)),
            'cashier': (int(frame_width * 0.4), int(frame_height * 0.75), int(frame_width * 0.6), frame_height),
            'exit': (int(frame_width * 0.8), 0, frame_width, frame_height)
        }
        self.zones_calibrated = True
        print("✅ Enhanced zones calibrated for pose analysis")
    
    def get_pose_keypoints(self, frame, bbox):
        """Extract pose keypoints for a specific person"""
        x1, y1, x2, y2 = bbox
        
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return None
            
        pose_results = self.pose_model(person_crop, verbose=False)
        
        if (pose_results[0].keypoints is not None and 
            len(pose_results[0].keypoints.data) > 0):
            keypoints = pose_results[0].keypoints.data[0]  
            
            adjusted_keypoints = keypoints.clone()
            adjusted_keypoints[:, 0] += x1  
            adjusted_keypoints[:, 1] += y1  
            
            return adjusted_keypoints.cpu().numpy()
        
        return None
    
    def detect_hand_movements(self, frame, bbox):
        """Detect detailed hand movements using MediaPipe"""
        x1, y1, x2, y2 = bbox
        
        padding = 30
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return None
            
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_crop)
        
        hand_data = {
            'left_hand': None,
            'right_hand': None,
            'hand_count': 0,
            'gestures': []
        }
        
        if hand_results.multi_hand_landmarks:
            hand_data['hand_count'] = len(hand_results.multi_hand_landmarks)
            
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                hand_label = hand_results.multi_handedness[idx].classification[0].label
                
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * person_crop.shape[1]) + x1
                    y = int(landmark.y * person_crop.shape[0]) + y1
                    landmarks.append((x, y))
                
                hand_data[f"{hand_label.lower()}_hand"] = landmarks
                
                gestures = self.analyze_hand_gestures(landmarks)
                hand_data['gestures'].extend(gestures)
        
        return hand_data
    
    def analyze_hand_gestures(self, landmarks):
        """Analyze hand landmarks for suspicious gestures"""
        if not landmarks or len(landmarks) < 21:
            return []
        
        gestures = []
        
        if self.is_closed_fist(landmarks):
            gestures.append('closed_fist')
        
        if self.is_pointing(landmarks):
            gestures.append('pointing')
        
        if self.is_grabbing_motion(landmarks):
            gestures.append('grabbing')
        
        return gestures
    
    def is_closed_fist(self, landmarks):
        """Detect closed fist gesture"""
        palm_center = landmarks[0]  
        fingertips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        
        distances = []
        for tip in fingertips:
            dist = math.sqrt((tip[0] - palm_center[0])**2 + (tip[1] - palm_center[1])**2)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        return avg_distance < 50  
    
    def is_pointing(self, landmarks):
        """Detect pointing gesture"""
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        
        index_extended = abs(index_tip[1] - index_pip[1]) > 15  
        other_fingers_curled = landmarks[12][1] > landmarks[10][1]
        
        return index_extended and other_fingers_curled
    
    def is_grabbing_motion(self, landmarks):
        """Detect grabbing motion"""
        thumb_tip, index_tip, middle_tip = landmarks[4], landmarks[8], landmarks[12]
        palm_center = landmarks[0]
        
        distances = []
        for tip in [thumb_tip, index_tip, middle_tip]:
            dist = math.sqrt((tip[0] - palm_center[0])**2 + (tip[1] - palm_center[1])**2)
            distances.append(dist)
        
        return all(d < 60 for d in distances)  
    
    def analyze_pose_for_suspicious_actions(self, track_id, keypoints, hand_data, current_time):
        """FIXED: More sensitive pose analysis for suspicious actions"""
        if keypoints is None:
            return
        
        track_data = self.track_history[track_id]
        pose_analysis = track_data['pose_analysis']
        action_scores = track_data['action_scores']
        
        track_data['pose_keypoints'].append(keypoints)
        
        def get_keypoint_safe(idx):
            if idx < len(keypoints) and keypoints[idx][2] > 0.3:  
                return keypoints[idx][:2]  
            return None
        
        left_shoulder = get_keypoint_safe(self.keypoint_indices['left_shoulder'])
        right_shoulder = get_keypoint_safe(self.keypoint_indices['right_shoulder'])
        left_elbow = get_keypoint_safe(self.keypoint_indices['left_elbow'])
        right_elbow = get_keypoint_safe(self.keypoint_indices['right_elbow'])
        left_wrist = get_keypoint_safe(self.keypoint_indices['left_wrist'])
        right_wrist = get_keypoint_safe(self.keypoint_indices['right_wrist'])
        nose = get_keypoint_safe(self.keypoint_indices['nose'])
        left_hip = get_keypoint_safe(self.keypoint_indices['left_hip'])
        right_hip = get_keypoint_safe(self.keypoint_indices['right_hip'])
        
        if (left_shoulder is None or right_shoulder is None or 
            left_hip is None or right_hip is None):
            return
        
        self.detect_pocket_reaching_improved(left_wrist, right_wrist, left_hip, right_hip, 
                                           left_shoulder, right_shoulder, action_scores, pose_analysis)
        
        self.detect_bag_concealment_improved(keypoints, hand_data, action_scores, current_time, pose_analysis)
        
        self.analyze_body_language_improved(keypoints, pose_analysis, action_scores, current_time)
        

        if (
            nose is not None and left_shoulder is not None and right_shoulder is not None
            and np.any(nose) and np.any(left_shoulder) and np.any(right_shoulder)
        ):

            self.analyze_head_movements_improved(nose, left_shoulder, right_shoulder, pose_analysis, action_scores)
        
        self.detect_rapid_movements_improved(keypoints, pose_analysis, action_scores)
        

        self.calculate_pose_based_score_improved(track_id)
    
    def detect_pocket_reaching_improved(self, left_wrist, right_wrist, left_hip, right_hip, 
                                      left_shoulder, right_shoulder, action_scores, pose_analysis):
        """FIXED: More sensitive pocket reaching detection"""
        if (left_wrist is None or right_wrist is None or 
            left_hip is None or right_hip is None):
            return
        
        left_pocket_zone = (left_hip[0] - 30, left_hip[0] + 20, left_hip[1] - 10, left_hip[1] + 50)
        right_pocket_zone = (right_hip[0] - 20, right_hip[0] + 30, right_hip[1] - 10, right_hip[1] + 50)

        
        left_in_pocket = (left_pocket_zone[0] <= left_wrist[0] <= left_pocket_zone[1] and
                         left_pocket_zone[2] <= left_wrist[1] <= left_pocket_zone[3])
        
        right_in_pocket = (right_pocket_zone[0] <= right_wrist[0] <= right_pocket_zone[1] and
                          right_pocket_zone[2] <= right_wrist[1] <= right_pocket_zone[3])
        
        if left_in_pocket or right_in_pocket:
            pose_analysis['pocket_interaction_frames'] += 1
            action_scores['reaching_to_pocket'] += 0.5 
        else:
            pose_analysis['pocket_interaction_frames'] = 0
        
        if pose_analysis['pocket_interaction_frames'] >= self.pose_thresholds['pocket_interaction_threshold']:
            action_scores['reaching_to_pocket'] += 2.0  
            print(f"🚨 DEBUG: Pocket interaction detected for {pose_analysis['pocket_interaction_frames']} frames")
        

        if left_shoulder is not None and right_shoulder is not None:
            torso_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            behind_torso_threshold = 40  
            
            left_behind = left_wrist is not None and left_wrist[0] < torso_center_x - behind_torso_threshold
            right_behind = right_wrist is not None and right_wrist[0] > torso_center_x + behind_torso_threshold
            
            if left_behind or right_behind:
                pose_analysis['bag_interaction_frames'] += 1
                action_scores['hiding_in_bag'] += 0.4  
                print(f"🚨 DEBUG: Bag interaction detected - hands behind torso")
    
    def detect_bag_concealment_improved(self, keypoints, hand_data, action_scores, current_time, pose_analysis):
        """FIXED: More sensitive bag concealment detection"""
        if hand_data is None:
            return
        
        closed_fists = hand_data['gestures'].count('closed_fist')
        if closed_fists > 0:
            action_scores['concealing_object'] += closed_fists * 0.3  
            print(f"🚨 DEBUG: Closed fist detected ({closed_fists} hands)")
        
        grabbing_motions = hand_data['gestures'].count('grabbing')
        if grabbing_motions > 0:
            action_scores['hand_near_products'] += grabbing_motions * 0.2 
            print(f"🚨 DEBUG: Grabbing motion detected ({grabbing_motions} hands)")
        
        left_wrist_kp = keypoints[self.keypoint_indices['left_wrist']]
        right_wrist_kp = keypoints[self.keypoint_indices['right_wrist']]
        left_shoulder_kp = keypoints[self.keypoint_indices['left_shoulder']]
        right_shoulder_kp = keypoints[self.keypoint_indices['right_shoulder']]
        
        valid_keypoints = (left_wrist_kp[2] > 0.3 and right_wrist_kp[2] > 0.3 and 
                          left_shoulder_kp[2] > 0.3 and right_shoulder_kp[2] > 0.3)
        
        if valid_keypoints:
            torso_width = abs(right_shoulder_kp[0] - left_shoulder_kp[0])
            

            hands_near_torso = (
                abs(left_wrist_kp[0] - left_shoulder_kp[0]) < torso_width * 0.4 or  
                abs(right_wrist_kp[0] - right_shoulder_kp[0]) < torso_width * 0.4
            )
            
            if hands_near_torso:
                action_scores['concealing_object'] += 0.15  
                print(f"🚨 DEBUG: Hands near torso detected")
    
    def analyze_head_movements_improved(self, nose, left_shoulder, right_shoulder, pose_analysis, action_scores):
        """FIXED: More sensitive head movement analysis"""
        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                          (left_shoulder[1] + right_shoulder[1]) / 2)
        
        head_offset = nose[0] - shoulder_center[0]
        pose_analysis['head_direction'].append(head_offset)
        
        if len(pose_analysis['head_direction']) >= 5:  
            head_directions = list(pose_analysis['head_direction'])
            direction_changes = 0
            
            for i in range(1, len(head_directions)):
                if abs(head_directions[i] - head_directions[i-1]) > 20:  
                    direction_changes += 1
            
            if direction_changes >= 2:  
                action_scores['looking_around'] += 0.3  
                print(f"🚨 DEBUG: Head turning detected ({direction_changes} changes)")
    
    def analyze_body_language_improved(self, keypoints, pose_analysis, action_scores, current_time):
        """FIXED: More sensitive body language analysis"""
        nose = keypoints[self.keypoint_indices['nose']]
        left_shoulder = keypoints[self.keypoint_indices['left_shoulder']]
        right_shoulder = keypoints[self.keypoint_indices['right_shoulder']]
        left_hip = keypoints[self.keypoint_indices['left_hip']]
        right_hip = keypoints[self.keypoint_indices['right_hip']]
        
        required_keypoints = [nose, left_shoulder, right_shoulder, left_hip, right_hip]
        valid_keypoints = all(kp is not None and len(kp) >= 3 and kp[2] > 0.3 
                             for kp in required_keypoints)
        
        if not valid_keypoints:
            return
        
        shoulder_vector = np.array([right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1]])
        hip_vector = np.array([right_hip[0] - left_hip[0], right_hip[1] - left_hip[1]])
        

        shoulder_angle = math.degrees(math.atan2(shoulder_vector[1], shoulder_vector[0]))
        pose_analysis['shoulder_angle_history'].append(shoulder_angle)
        
        if len(pose_analysis['shoulder_angle_history']) > 3: 
            shoulder_angles = list(pose_analysis['shoulder_angle_history'])
            angle_variance = np.var(shoulder_angles)
            
            if angle_variance > 50: 
                action_scores['suspicious_body_language'] += 0.4 
                print(f"🚨 DEBUG: Suspicious body language - angle variance: {angle_variance:.1f}")

        shoulder_hip_distance = abs(left_shoulder[1] - left_hip[1])
        if shoulder_hip_distance < 100:  
            pose_analysis['crouching_detected'] = True
            action_scores['suspicious_body_language'] += 0.6  
            print(f"🚨 DEBUG: Crouching detected - distance: {shoulder_hip_distance}")
    
    def detect_rapid_movements_improved(self, keypoints, pose_analysis, action_scores):
        """FIXED: More sensitive rapid movement detection"""
        left_wrist = keypoints[self.keypoint_indices['left_wrist']]
        right_wrist = keypoints[self.keypoint_indices['right_wrist']]
        
        if left_wrist[2] < 0.3 or right_wrist[2] < 0.3:
            return
        
        if len(pose_analysis['hand_movement_speed']) > 0:
            prev_data = pose_analysis['hand_movement_speed'][-1]
            prev_left_wrist = prev_data['left_wrist']
            prev_right_wrist = prev_data['right_wrist']
            
            left_speed = math.sqrt((left_wrist[0] - prev_left_wrist[0])**2 + 
                                  (left_wrist[1] - prev_left_wrist[1])**2)
            right_speed = math.sqrt((right_wrist[0] - prev_right_wrist[0])**2 + 
                                   (right_wrist[1] - prev_right_wrist[1])**2)
            
            threshold = self.pose_thresholds['rapid_movement_threshold']  
            if left_speed > threshold:
                action_scores['rapid_movements'] += 0.4  
                print(f"🚨 DEBUG: Rapid left hand movement: {left_speed:.1f} pixels")
            if right_speed > threshold:
                action_scores['rapid_movements'] += 0.4  
                print(f"🚨 DEBUG: Rapid right hand movement: {right_speed:.1f} pixels")
        

        pose_analysis['hand_movement_speed'].append({
            'left_wrist': left_wrist[:2],  
            'right_wrist': right_wrist[:2]
        })
    
    def calculate_pose_based_score_improved(self, track_id):
        """FIXED: More sensitive scoring system"""
        track_data = self.track_history[track_id]
        action_scores = track_data['action_scores']
        
        total_score = 0
        significant_actions = 0
        
        min_thresholds = {
            'reaching_to_pocket': 12.0,
            'hiding_in_bag': 12.0,
            'concealing_object': 12.0,
            'looking_around': 12.0,
            'rapid_movements': 12.0,
            'hand_near_products': 12.0,
            'suspicious_body_language': 12.0
        }

        for action, score in action_scores.items():
            weight = self.action_weights.get(action, 1.0)
            threshold = min_thresholds.get(action, 1.0)
            
            if score >= threshold:
                significant_actions += 1
                weighted_score = score * weight  
                total_score += min(weighted_score, 12.0)  
                print(f"🚨 DEBUG: {action}: score={score:.1f}, weight={weight}, contribution={min(weighted_score, 12.0):.1f}")
        
        if significant_actions < 1:
            total_score *= 0.8 
        
        current_time = time.time()
        time_since_last = current_time - track_data['last_action_time']
        
        if time_since_last > 5: 
            for action in action_scores:
                if action not in ['hiding_in_bag', 'concealing_object']:
                    action_scores[action] = max(0, action_scores[action] - 0.1)  
                else:
                    action_scores[action] = max(0, action_scores[action] - 0.05)  
        
        track_data['last_action_time'] = current_time
        track_data['suspicious_score'] = min(total_score, 120.0) 
        
        if track_data['suspicious_score'] >= 60:
            track_data['alert_level'] = 'CRITICAL'
        elif track_data['suspicious_score'] >= 50:
            track_data['alert_level'] = 'HIGH'
        elif track_data['suspicious_score'] >= 45:  
            track_data['alert_level'] = 'MEDIUM'
        else:
            track_data['alert_level'] = 'LOW'
        
        if track_data['suspicious_score'] > 2:
            print(f"🚨 DEBUG: Track {track_id} - Score: {track_data['suspicious_score']:.1f}, "
                  f"Alert: {track_data['alert_level']}, Actions: {significant_actions}")
    
    def draw_pose_analysis(self, frame, keypoints, track_id, bbox):
        """Draw pose keypoints and analysis results"""
        if keypoints is None:
            return
        
        x1, y1, x2, y2 = bbox
        track_data = self.track_history[track_id]
        
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3: 
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), 
            (5, 11), (6, 12), (11, 12),  
            (11, 13), (13, 15), (12, 14), (14, 16) 
        ]
        
        for start_idx, end_idx in connections:
            if (keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        
        left_hip_kp = keypoints[self.keypoint_indices['left_hip']]
        right_hip_kp = keypoints[self.keypoint_indices['right_hip']]
        
        if left_hip_kp[2] > 0.3 and right_hip_kp[2] > 0.3:
            left_pocket = (int(left_hip_kp[0] - 60), int(left_hip_kp[1] - 20), 
                          int(left_hip_kp[0] + 30), int(left_hip_kp[1] + 80))
            cv2.rectangle(frame, (left_pocket[0], left_pocket[1]), 
                         (left_pocket[2], left_pocket[3]), (255, 255, 0), 1)
            
            right_pocket = (int(right_hip_kp[0] - 30), int(right_hip_kp[1] - 20), 
                           int(right_hip_kp[0] + 60), int(right_hip_kp[1] + 80))
            cv2.rectangle(frame, (right_pocket[0], right_pocket[1]), 
                         (right_pocket[2], right_pocket[3]), (255, 255, 0), 1)
        
        info_y = y1 - 100
        
        alert_color = (0, 0, 255) if track_data['alert_level'] == 'CRITICAL' else \
                     (0, 165, 255) if track_data['alert_level'] == 'HIGH' else \
                     (0, 255, 255) if track_data['alert_level'] == 'MEDIUM' else (0, 255, 0)
        
    def process_frame(self, frame):
        """Process frame with improved pose-based detection"""
        self.frame_count += 1
        current_time = time.time()
        

        if not self.zones_calibrated:
            h, w = frame.shape[:2]
            self.auto_calibrate_zones(w, h)
        
 
        detection_results = self.detection_model.track(frame, persist=True, classes=0)
        
        suspicious_persons = []
        critical_alerts = []
        
        if detection_results[0].boxes is not None and detection_results[0].boxes.id is not None:
            boxes = detection_results[0].boxes.xyxy.int().cpu().tolist()
            track_ids = detection_results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                
                keypoints = self.get_pose_keypoints(frame, box)
                
                hand_data = self.detect_hand_movements(frame, box)
                
                if hand_data:
                    self.track_history[track_id]['hand_positions'].append(hand_data)
                
                self.analyze_pose_for_suspicious_actions(track_id, keypoints, hand_data, current_time)
                
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                self.track_history[track_id]['positions'].append((center_x, center_y))
                self.track_history[track_id]['timestamps'].append(current_time)

                track_data = self.track_history[track_id]
                alert_level = track_data['alert_level']
                score = track_data['suspicious_score']
                
                if alert_level in ['HIGH', 'CRITICAL']:
                    suspicious_persons.append(track_id)
                    
                if alert_level == 'CRITICAL':
                    critical_alerts.append({
                        'track_id': track_id,
                        'score': score,
                        'actions': [k for k, v in track_data['action_scores'].items() if v > 1.0],
                        'timestamp': current_time
                    })
                
                colors = {
                    'LOW': (0, 255, 0),      
                    'MEDIUM': (0, 255, 255),  
                    'HIGH': (0, 165, 255),    
                    'CRITICAL': (0, 0, 255)   #
                }
                
                color = colors[alert_level]
                thickness = 2 if alert_level == 'LOW' else 5 if alert_level == 'CRITICAL' else 3
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                self.draw_pose_analysis(frame, keypoints, track_id, box)
                
                cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y1-80), 1, 2)
                cvzone.putTextRect(frame, f'{alert_level} RISK', (x1, y1-55), 1, 2, colorR=color)
        
        return frame, suspicious_persons, critical_alerts

def main():
    print("Starting Improved Pose-Based Shoplifting Detection System v3.1...")
    
    detector = PoseShopliftingDetector()
    
    video_path = input("Enter video file path (or press Enter for 'susp1.mp4'): ").strip()
    if not video_path:
        video_path = 'input1.mp4'
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return []
    
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (min(1280, w), int(min(1280, w) * h / w))
    
    alert_log = []
    critical_alert_count = 0
    suspicious_alert_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, frame_size)
        processed_frame, suspicious_persons, critical_alerts = detector.process_frame(frame)
        
        # Simpan critical alerts ke log
        if critical_alerts:
            critical_alert_count += len(critical_alerts)
            for alert in critical_alerts:
                alert_info = {
                    'frame': detector.frame_count,
                    'track_id': alert['track_id'],
                    'score': alert['score'],
                    'suspicious_actions': alert['actions'],
                    'timestamp': alert['timestamp'],
                    'alert_type': 'CRITICAL'
                }
                alert_log.append(alert_info)
        
        if suspicious_persons:
            suspicious_alert_count += len(suspicious_persons)
        
        # Tampilkan progress
        progress = (detector.frame_count / total_frames) * 100
        cv2.putText(processed_frame, f"Progress: {progress:.1f}%", 
                   (frame_size[0] - 200, frame_size[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow("Improved Pose-Based Shoplifting Detection v3.1", processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Simpan ke file JSON
    if alert_log:
        log_path = f"improved_alert_log_{int(time.time())}.json"
        with open(log_path, 'w') as f:
            json.dump(alert_log, f, indent=2, default=str)
        print(f"Alert log saved: {log_path}")
    
    print(f"Critical alerts: {critical_alert_count}")
    print(f"Suspicious alerts: {suspicious_alert_count}")
    
    # ✅ return alert_log sebagai list of dict
    return alert_log


if __name__ == "__main__":
    results = main()
    print("\nReturned results (list of dict):")
    print(json.dumps(results[:5], indent=2, default=str))  # tampilkan max 5 contoh
