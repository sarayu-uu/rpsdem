import cv2
import numpy as np
import math

class HandDetector:
    def __init__(self):
        # Parameters for skin color detection - wider range for better detection
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([30, 255, 255], dtype=np.uint8)
        
        # Alternative skin color ranges for different lighting conditions
        self.skin_ranges = [
            # Standard range
            (np.array([0, 20, 70], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8)),
            # For darker skin tones
            (np.array([0, 10, 60], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8)),
            # For brighter lighting
            (np.array([0, 30, 80], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8))
        ]
        self.current_skin_range = 0  # Index of the current skin range
        
        # Parameters for gesture recognition
        self.gesture_history = []
        self.history_size = 10  # Increased for more stability
        self.min_contour_area = 3000  # Reduced to detect smaller hand contours
        
        # ROI (Region of Interest) for hand detection
        self.roi_top = 100
        self.roi_bottom = 400
        self.roi_left = 100
        self.roi_right = 400
        
        # Debug mode
        self.debug = True
        
        # Adaptive parameters
        self.adaptive_mode = True  # Enable adaptive skin detection
        self.frame_count = 0
        self.detection_success_rate = 0
        self.last_successful_range = 0
    
    def set_roi(self, top, bottom, left, right):
        """Set the region of interest for hand detection"""
        self.roi_top = top
        self.roi_bottom = bottom
        self.roi_left = left
        self.roi_right = right
    
    def detect_hands(self, frame):
        """
        Detect hands in the frame using skin color segmentation
        Returns: processed frame and contour of the hand if detected
        """
        try:
            # Create a copy of the frame for drawing
            display_frame = frame.copy()
            
            # Extract the region of interest
            roi = frame[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]
            
            # Draw ROI rectangle
            cv2.rectangle(display_frame, (self.roi_left, self.roi_top), 
                         (self.roi_right, self.roi_bottom), (0, 255, 0), 2)
            
            # Check if ROI is valid
            if roi.size == 0:
                return display_frame, None
            
            # Convert to HSV color space
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Increment frame count for adaptive mode
            self.frame_count += 1
            
            # Try different skin color ranges if in adaptive mode
            if self.adaptive_mode and self.frame_count % 30 == 0:  # Every 30 frames
                # If detection rate is low, try a different range
                if self.detection_success_rate < 0.3:  # Less than 30% success
                    self.current_skin_range = (self.current_skin_range + 1) % len(self.skin_ranges)
                    print(f"Switching to skin range {self.current_skin_range}")
                    self.detection_success_rate = 0
            
            # Get current skin color range
            self.lower_skin, self.upper_skin = self.skin_ranges[self.current_skin_range]
            
            # Create a mask for skin color
            mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Apply additional filtering to improve detection
            # Use background subtraction or motion detection if needed
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # If debug mode is on, show the mask and current skin range
            if self.debug:
                # Resize the mask for display
                mask_display = cv2.resize(mask, (200, 200))
                # Convert to BGR for display
                mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
                # Place it in the top-right corner
                display_frame[10:210, display_frame.shape[1]-210:display_frame.shape[1]-10] = mask_display
                
                # Display current skin range
                cv2.putText(display_frame, f"Skin Range: {self.current_skin_range}", 
                           (display_frame.shape[1]-210, 230), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
            
            # Find the largest contour (assumed to be the hand)
            if contours:
                # Sort contours by area (largest first)
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Filter contours by shape and size
                valid_contours = []
                for contour in sorted_contours:
                    area = cv2.contourArea(contour)
                    if area > self.min_contour_area:
                        # Calculate contour properties
                        perimeter = cv2.arcLength(contour, True)
                        # Contour must have a reasonable perimeter-to-area ratio
                        if perimeter > 0 and area / perimeter > 10:
                            valid_contours.append(contour)
                
                if valid_contours:
                    max_contour = valid_contours[0]
                    
                    # Update detection success rate
                    self.detection_success_rate = 0.7 * self.detection_success_rate + 0.3
                    self.last_successful_range = self.current_skin_range
                    
                    # Draw the contour on the display frame
                    offset_contour = []
                    for point in max_contour:
                        offset_point = [point[0][0] + self.roi_left, point[0][1] + self.roi_top]
                        offset_contour.append([offset_point])
                    
                    offset_contour = np.array(offset_contour, dtype=np.int32)
                    cv2.drawContours(display_frame, [offset_contour], 0, (0, 0, 255), 2)
                    
                    # Create a mask of the largest contour for gesture analysis
                    hand_mask = np.zeros_like(mask)
                    cv2.drawContours(hand_mask, [max_contour], 0, 255, -1)
                    
                    # Draw convex hull for better visualization
                    hull = cv2.convexHull(max_contour)
                    hull_offset = []
                    for point in hull:
                        hull_point = [point[0][0] + self.roi_left, point[0][1] + self.roi_top]
                        hull_offset.append([hull_point])
                    
                    hull_offset = np.array(hull_offset, dtype=np.int32)
                    cv2.drawContours(display_frame, [hull_offset], 0, (0, 255, 255), 2)
                    
                    return display_frame, max_contour
                else:
                    # Update detection success rate (failure)
                    self.detection_success_rate = 0.7 * self.detection_success_rate
        
        except Exception as e:
            print(f"Error in detect_hands: {e}")
        
        return display_frame, None
    
    def get_gesture(self, contour):
        """
        Determine the gesture (rock, paper, scissors) based on contour analysis
        Returns: string representing the detected gesture
        
        Standard gestures:
        - Rock: Closed fist (all fingers folded)
        - Paper: Open palm with all fingers extended
        - Scissors: Only index and middle fingers extended, others folded
        """
        if contour is None:
            return None
        
        try:
            # Calculate convex hull
            hull = cv2.convexHull(contour, returnPoints=False)
            hull_points = cv2.convexHull(contour, returnPoints=True)
            
            # If hull is too small, can't compute defects
            if len(hull) < 3:
                return "rock"  # Default to rock if we can't analyze properly
            
            # Calculate convexity defects
            defects = cv2.convexityDefects(contour, hull)
            
            # Calculate contour properties for better classification
            area = cv2.contourArea(contour)
            hull_area = cv2.contourArea(hull_points)
            perimeter = cv2.arcLength(contour, True)
            
            # Solidity = contour area / convex hull area
            # Rock: high solidity (close to 1) - compact shape with no protruding fingers
            # Paper: lower solidity - open palm with fingers creates more convexity defects
            # Scissors: medium solidity - two fingers create some defects
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Circularity = 4*pi*area / perimeter^2 (1.0 for perfect circle)
            # Rock tends to be more circular
            circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Count the number of extended fingers (not including palm)
            extended_fingers = 0
            finger_tips = []
            
            if defects is not None and defects.shape[0] > 0:
                # For each defect point, we analyze the angle
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    
                    # Calculate the triangle sides
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    
                    # Avoid division by zero
                    if b * c == 0:
                        continue
                    
                    # Calculate the angle between fingers
                    try:
                        angle = math.degrees(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
                        
                        # If the angle is less than 90 degrees, it's likely a finger valley
                        if angle <= 90:
                            # Store the fingertip coordinates (end point)
                            finger_tips.append(end)
                            extended_fingers += 1
                    except:
                        # Skip this defect if there's a math error
                        continue
                
                # For most hand poses, we need to add 1 to account for the last finger
                # that doesn't create a defect with the next finger
                if extended_fingers > 0:
                    extended_fingers += 1
            
            # Special case for scissors - check if exactly two fingers are extended
            # and they are close together (index and middle finger)
            is_scissors = False
            if extended_fingers == 2 and len(finger_tips) >= 1:
                # For scissors, we expect two fingers close together
                # and the shape should not be too circular
                if circularity < 0.7:
                    is_scissors = True
            
            # Add current finger count to history for stability
            self.gesture_history.append(extended_fingers)
            if len(self.gesture_history) > self.history_size:
                self.gesture_history.pop(0)
            
            # Get the most common finger count from history
            if self.gesture_history:
                counts = {}
                for count in self.gesture_history:
                    if count in counts:
                        counts[count] += 1
                    else:
                        counts[count] = 1
                
                extended_fingers = max(counts, key=counts.get)
            
            # Print debug info
            print(f"Fingers: {extended_fingers}, Solidity: {solidity:.2f}, Circularity: {circularity:.2f}")
            
            # Determine gesture based on finger count, solidity and circularity
            if extended_fingers <= 1 or (circularity > 0.7 and solidity > 0.8):
                # Rock: closed fist (0-1 fingers or very circular and solid shape)
                return "rock"
            elif extended_fingers == 2 or is_scissors:
                # Scissors: exactly 2 fingers (index and middle)
                return "scissors"
            elif extended_fingers >= 4 or (extended_fingers >= 3 and solidity < 0.75):
                # Paper: all fingers extended (4-5 fingers or 3+ with low solidity)
                return "paper"
            else:
                # Default to rock for ambiguous cases
                return "rock"
                
        except Exception as e:
            print(f"Error in get_gesture: {e}")
            return None
    
    def draw_gesture_info(self, frame, gesture):
        """
        Draw gesture information on the frame
        """
        if gesture:
            # Draw the detected gesture name with a background
            text = f"Detected: {gesture.upper()}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, 
                         (10, frame.shape[0] - 70), 
                         (10 + text_size[0] + 10, frame.shape[0] - 40), 
                         (0, 0, 0), -1)
            cv2.putText(frame, text, 
                       (15, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2)
            
            # Draw a visual representation of the gesture
            icon_size = 100
            icon_margin = 10
            icon_position = (frame.shape[1] - icon_size - icon_margin, 
                            frame.shape[0] - icon_size - icon_margin)
            
            # Create a background for the icon
            cv2.rectangle(frame, 
                         icon_position, 
                         (icon_position[0] + icon_size, icon_position[1] + icon_size), 
                         (50, 50, 50), -1)
            
            # Draw the gesture icon
            if gesture == "rock":
                # Draw a fist for rock (circle with smaller circles inside)
                center = (icon_position[0] + icon_size // 2, icon_position[1] + icon_size // 2)
                # Main fist
                cv2.circle(frame, center, icon_size // 3, (0, 0, 255), -1)
                # Knuckles
                for i in range(4):
                    knuckle_x = center[0] - icon_size//6 + (i * icon_size//12)
                    knuckle_y = center[1] - icon_size//8
                    cv2.circle(frame, (knuckle_x, knuckle_y), icon_size//12, (150, 0, 0), -1)
                
                # Add text label
                cv2.putText(frame, "ROCK", 
                           (icon_position[0] + 10, icon_position[1] + icon_size - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            elif gesture == "paper":
                # Draw an open hand for paper (palm with fingers)
                # Palm
                palm_center = (icon_position[0] + icon_size // 2, icon_position[1] + icon_size // 2 + 10)
                cv2.circle(frame, palm_center, icon_size // 4, (255, 0, 0), -1)
                
                # Fingers
                for i in range(5):
                    angle = math.pi/2 - (i * math.pi/5)  # Spread fingers in a fan
                    finger_length = icon_size // 3
                    finger_x = int(palm_center[0] + finger_length * math.cos(angle))
                    finger_y = int(palm_center[1] - finger_length * math.sin(angle))
                    cv2.line(frame, palm_center, (finger_x, finger_y), (255, 0, 0), 5)
                    cv2.circle(frame, (finger_x, finger_y), 5, (200, 0, 0), -1)
                
                # Add text label
                cv2.putText(frame, "PAPER", 
                           (icon_position[0] + 10, icon_position[1] + icon_size - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            elif gesture == "scissors":
                # Draw scissors (palm with two extended fingers)
                # Palm
                palm_center = (icon_position[0] + icon_size // 2, icon_position[1] + icon_size // 2 + 15)
                cv2.circle(frame, palm_center, icon_size // 5, (0, 200, 0), -1)
                
                # Two extended fingers (index and middle)
                angle1 = math.pi/2 + math.pi/12  # Slightly to the left
                angle2 = math.pi/2 - math.pi/12  # Slightly to the right
                finger_length = icon_size // 2.5
                
                # Index finger
                finger1_x = int(palm_center[0] + finger_length * math.cos(angle1))
                finger1_y = int(palm_center[1] - finger_length * math.sin(angle1))
                cv2.line(frame, palm_center, (finger1_x, finger1_y), (0, 255, 0), 5)
                cv2.circle(frame, (finger1_x, finger1_y), 5, (0, 150, 0), -1)
                
                # Middle finger
                finger2_x = int(palm_center[0] + finger_length * math.cos(angle2))
                finger2_y = int(palm_center[1] - finger_length * math.sin(angle2))
                cv2.line(frame, palm_center, (finger2_x, finger2_y), (0, 255, 0), 5)
                cv2.circle(frame, (finger2_x, finger2_y), 5, (0, 150, 0), -1)
                
                # Add text label
                cv2.putText(frame, "SCISSORS", 
                           (icon_position[0] + 5, icon_position[1] + icon_size - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add gesture tips
            tips = {
                "rock": "Make a fist (close all fingers)",
                "paper": "Open palm with all fingers extended",
                "scissors": "Extend only index and middle fingers"
            }
            
            tip_text = f"Tip: {tips[gesture]}"
            tip_size = cv2.getTextSize(tip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, 
                         (10, frame.shape[0] - 100), 
                         (10 + tip_size[0] + 10, frame.shape[0] - 75), 
                         (0, 0, 0), -1)
            cv2.putText(frame, tip_text, 
                       (15, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 200, 200), 1)
        
        return frame