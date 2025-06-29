import cv2
import numpy as np
import random
import time
import os
import math
import json
import datetime
import csv
import winsound  # For sound effects on Windows
from hand_detector import HandDetector
from collections import deque, Counter

class RockPaperScissorsGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        # Set a fixed window size to prevent text from being cut off
        self.width = 800
        self.height = 600
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.choices = ['rock', 'paper', 'scissors']
        self.user_choice = None
        self.computer_choice = None
        self.result = None
        self.countdown = 3
        self.last_countdown_time = time.time()
        self.game_state = "countdown"  # States: countdown, detection, result, menu
        self.result_display_time = None
        self.score = {"user": 0, "computer": 0}
        self.rounds_played = 0
        self.max_rounds = 5  # Best of 5 rounds by default
        
        # Game UI colors
        self.bg_color = (0, 0, 0)  # Black
        self.title_color = (0, 255, 255)  # Yellow
        self.score_color = (0, 255, 0)  # Green
        self.countdown_color = (0, 0, 255)  # Red
        self.instruction_color = (255, 255, 255)  # White
        self.result_color = (255, 165, 0)  # Orange
        self.menu_color = (200, 200, 255)  # Light blue
        
        # Initialize hand detector
        self.hand_detector = HandDetector()
        
        # Gesture detection variables
        self.detection_start_time = None
        self.detection_duration = 3  # seconds to detect gesture
        self.detected_gestures = []
        self.gesture_confidence_threshold = 3  # Number of consistent detections needed
        
        # Animation variables
        self.animation_frame = 0
        self.animation_speed = 5  # frames per animation step
        self.frame_count = 0
        
        # Sound effect frequencies (in Hz)
        self.sound_effects = {
            "countdown": 440,  # A4 note
            "win": 880,        # A5 note
            "lose": 220,       # A3 note
            "tie": 660,        # E5 note
            "error": 110       # A2 note
        }
        
        # Gesture detection feedback
        self.detection_feedback = ""
        self.feedback_color = (255, 255, 255)
        self.no_hand_count = 0
        self.max_no_hand_count = 10  # Number of frames without hand before showing warning
        
        # Game history and AI strategy variables
        self.game_history = []
        self.player_move_history = deque(maxlen=10)  # Store last 10 moves
        self.ai_mode = "random"  # Modes: random, pattern, counter
        self.ai_difficulty = 0.7  # Probability of using strategy vs random (0.0 to 1.0)
        
        # Create logs directory if it doesn't exist
        self.logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            
        # Initialize log file with session info
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.logs_dir, f"game_log_{self.session_id}.json")
        self.initialize_log_file()
        
        # Message display variables
        self.display_message = ""
        self.display_message_time = 0
        self.display_message_duration = 0
        
    def initialize_log_file(self):
        """Initialize the log file with session information"""
        session_info = {
            "session_id": self.session_id,
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "game_version": "1.0",
            "rounds": []
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(session_info, f, indent=4)
            
    def log_round(self, round_data):
        """Log a round to the JSON file"""
        try:
            # Read existing data
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
            
            # Add new round data
            log_data["rounds"].append(round_data)
            
            # Write updated data
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=4)
                
            # Also update the CSV export after each round
            self.export_to_csv()
                
        except Exception as e:
            print(f"Error logging round: {e}")
            
    def export_to_csv(self):
        """Export game history to a CSV file for easier analysis"""
        try:
            # Create CSV file path
            csv_file = os.path.join(self.logs_dir, f"game_stats_{self.session_id}.csv")
            
            # Define CSV columns based on round data structure
            fieldnames = [
                "round_number", "timestamp", "user_choice", "computer_choice", 
                "winner", "ai_mode", "ai_difficulty", "user_score", "computer_score",
                "decision_time", "current_streak", "streak_type", "previous_user_move",
                "detection_duration", "match_progress"
            ]
            
            # Write data to CSV
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write each round as a row
                for round_data in self.game_history:
                    # Filter to include only the defined fields
                    filtered_data = {k: round_data.get(k, '') for k in fieldnames}
                    writer.writerow(filtered_data)
                    
            return csv_file
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return None
            
    def get_computer_choice(self):
        """Get computer's choice based on AI strategy"""
        # Store the win/loss history for adaptive learning
        win_streak = 0
        loss_streak = 0
        
        # Calculate current streaks to adjust strategy
        if len(self.game_history) >= 3:
            recent_results = [round.get("winner") for round in self.game_history[-3:]]
            if all(result == "computer" for result in recent_results):
                win_streak = len(recent_results)
            elif all(result == "user" for result in recent_results):
                loss_streak = len(recent_results)
        
        # If we don't have enough history or using random mode, choose randomly
        if len(self.player_move_history) < 3 or self.ai_mode == "random" or random.random() > self.ai_difficulty:
            return random.choice(self.choices)
        
        # Pattern recognition strategy
        if self.ai_mode == "pattern":
            return self.pattern_based_choice()
        
        # Counter strategy (choose move that beats player's most frequent move)
        if self.ai_mode == "counter":
            return self.counter_strategy_choice()
            
        # Adaptive strategy - combines different approaches based on game state
        if self.ai_mode == "adaptive":
            # If we're on a winning streak, stick with what's working
            if win_streak >= 2:
                # 70% chance to use the same strategy that's been winning
                if random.random() < 0.7:
                    last_round = self.game_history[-1]
                    computer_last_move = last_round.get("computer_choice")
                    user_last_move = last_round.get("user_choice")
                    
                    # Predict if player will change strategy after losing
                    if random.random() < 0.6:  # Players often change after losing
                        # Counter what would beat their previous move
                        counter_to_expected = self.get_winning_move_against(
                            self.get_winning_move_against(user_last_move)
                        )
                        return counter_to_expected
                    else:
                        # Keep using the same winning move
                        return computer_last_move
            
            # If we're on a losing streak, try a different approach
            if loss_streak >= 2:
                # Mix between pattern and counter strategies
                if random.random() < 0.5:
                    return self.pattern_based_choice()
                else:
                    # Try a completely random move to break pattern
                    return random.choice(self.choices)
            
            # No streak - use weighted combination of strategies
            strategies = [
                self.pattern_based_choice,
                self.counter_strategy_choice,
                lambda: random.choice(self.choices)
            ]
            
            # Weights favor pattern recognition and counter strategies
            weights = [0.5, 0.3, 0.2]
            
            # Choose strategy based on weights
            strategy_index = random.choices(range(len(strategies)), weights=weights)[0]
            return strategies[strategy_index]()
        
        # Fallback to random
        return random.choice(self.choices)
        
    def pattern_based_choice(self):
        """Use pattern recognition to predict player's next move"""
        history_list = list(self.player_move_history)
        
        # If we don't have enough history, use random choice
        if len(history_list) < 3:
            return random.choice(self.choices)
            
        # Advanced pattern recognition with confidence scoring
        predictions = []
        confidence_scores = []
        
        # 1. Check for repeating moves (e.g., rock, rock, rock)
        last_three = history_list[-3:]
        if len(set(last_three)) == 1:
            # Player repeated same move 3 times, they might continue
            predictions.append(last_three[0])
            # Higher confidence if they've done this more than once
            repeat_patterns = 0
            for i in range(0, len(history_list) - 3, 3):
                if len(set(history_list[i:i+3])) == 1:
                    repeat_patterns += 1
            confidence_scores.append(0.7 + (0.1 * min(repeat_patterns, 3)))
        
        # 2. Check for alternating patterns (e.g., rock-paper-rock-paper)
        if len(history_list) >= 4:
            last_four = history_list[-4:]
            if last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                # Predict next move in alternating pattern will be the same as first move
                predictions.append(last_four[0])
                confidence_scores.append(0.8)
                
            # Check for three-move patterns (rock-paper-scissors-rock-paper-scissors)
            if len(history_list) >= 6:
                last_six = history_list[-6:]
                if last_six[0] == last_six[3] and last_six[1] == last_six[4] and last_six[2] == last_six[5]:
                    predictions.append(last_six[0])
                    confidence_scores.append(0.85)
        
        # 3. Look for sequence patterns (what follows after specific sequences)
        for seq_length in range(2, min(5, len(history_list) - 1)):
            current_seq = tuple(history_list[-seq_length:])
            
            # Find all instances of this sequence in history
            sequence_followers = []
            for i in range(len(history_list) - seq_length):
                if tuple(history_list[i:i+seq_length]) == current_seq:
                    if i + seq_length < len(history_list):
                        sequence_followers.append(history_list[i+seq_length])
            
            # If we found this sequence before, predict the most likely next move
            if sequence_followers:
                follower_counts = Counter(sequence_followers)
                most_common = follower_counts.most_common(1)[0]
                predicted_move = most_common[0]
                # Confidence based on how often this pattern occurred and the sequence length
                pattern_confidence = (most_common[1] / len(sequence_followers)) * (seq_length / 4)
                predictions.append(predicted_move)
                confidence_scores.append(min(0.9, 0.6 + pattern_confidence))
        
        # 4. Check for move cycling (player tends to cycle through moves)
        if len(set(history_list[-5:])) == 3 and len(history_list) >= 5:
            # Player has used all three moves recently, check if they avoid repeating
            last_move = history_list[-1]
            second_last = history_list[-2]
            # Predict they'll use the move they haven't used in the last two moves
            unused_move = [move for move in self.choices if move != last_move and move != second_last]
            if unused_move:
                predictions.append(unused_move[0])
                confidence_scores.append(0.6)
        
        # 5. Psychological pattern: players often change after losing
        if len(self.game_history) >= 2:
            last_round = self.game_history[-1]
            if last_round.get("winner") == "computer":
                # Player lost last round, they might change strategy
                last_move = last_round.get("user_choice")
                if last_move:
                    # Predict they'll switch to the move that would have beaten the computer's last move
                    computer_last_move = last_round.get("computer_choice")
                    if computer_last_move:
                        predicted_move = self.get_winning_move_against(computer_last_move)
                        predictions.append(predicted_move)
                        confidence_scores.append(0.65)
        
        # Make final decision based on predictions and confidence
        if predictions:
            # If we have multiple predictions, use the one with highest confidence
            if len(predictions) > 1:
                best_prediction_index = confidence_scores.index(max(confidence_scores))
                predicted_move = predictions[best_prediction_index]
                return self.get_winning_move_against(predicted_move)
            else:
                return self.get_winning_move_against(predictions[0])
        
        # Fallback to counter strategy if pattern recognition fails
        return self.counter_strategy_choice()
    
    def counter_strategy_choice(self):
        """Choose move that beats player's most frequent move with weighted recency"""
        # If we don't have enough history, use random choice
        if len(self.player_move_history) < 2:
            return random.choice(self.choices)
            
        # Count frequency of each move with recency weighting
        history_list = list(self.player_move_history)
        move_weights = {'rock': 0, 'paper': 0, 'scissors': 0}
        
        # Apply recency weighting - more recent moves count more
        for i, move in enumerate(history_list):
            # Exponential recency weight: more recent moves have higher weight
            recency_weight = 1.0 + (0.1 * i)  # Older moves get higher index
            move_weights[move] += recency_weight
            
        # Check for recent trends (last 3 moves)
        if len(history_list) >= 3:
            recent_moves = history_list[-3:]
            recent_counts = Counter(recent_moves)
            
            # If player is showing a strong recent preference, increase its weight
            for move, count in recent_counts.items():
                if count >= 2:  # They used this move at least twice in last 3 moves
                    move_weights[move] += 2.0
        
        # Check if player tends to avoid a particular move
        if len(history_list) >= 5:
            last_five = history_list[-5:]
            five_counts = Counter(last_five)
            
            # If a move is rarely used, they might be more likely to use it next
            for move in self.choices:
                if five_counts.get(move, 0) == 0:  # Move not used in last 5 rounds
                    # They might use this unused move, so counter it
                    counter_move = self.get_winning_move_against(move)
                    return counter_move
        
        # Find the move with highest weight
        if move_weights:
            most_common_move = max(move_weights.items(), key=lambda x: x[1])[0]
            return self.get_winning_move_against(most_common_move)
        
        # Fallback to random
        return random.choice(self.choices)
    
    def get_winning_move_against(self, move):
        """Return the move that beats the given move"""
        if move == 'rock':
            return 'paper'
        elif move == 'paper':
            return 'scissors'
        elif move == 'scissors':
            return 'rock'
        else:
            return random.choice(self.choices)
    
    def determine_winner(self, user_choice, computer_choice):
        self.rounds_played += 1
        round_start_time = time.time()
        
        # Add user's choice to move history for AI learning
        self.player_move_history.append(user_choice)
        
        # Determine the winner
        if user_choice == computer_choice:
            result = "It's a tie!"
            winner = "tie"
            self.play_sound("tie")
        elif (user_choice == 'rock' and computer_choice == 'scissors') or \
             (user_choice == 'paper' and computer_choice == 'rock') or \
             (user_choice == 'scissors' and computer_choice == 'paper'):
            result = "You win!"
            winner = "user"
            self.score["user"] += 1
            self.play_sound("win")
        else:
            result = "Computer wins!"
            winner = "computer"
            self.score["computer"] += 1
            self.play_sound("lose")
        
        # Calculate decision time (time from detection to result)
        decision_time = round(time.time() - round_start_time, 2)
        
        # Calculate streaks
        current_streak = 1
        streak_type = winner
        
        if self.game_history:
            last_round = self.game_history[-1]
            last_winner = last_round.get("winner")
            
            if last_winner == winner:
                # Continue the streak
                current_streak = last_round.get("current_streak", 0) + 1
        
        # Calculate move transitions (what move follows what)
        previous_move = None
        if len(self.player_move_history) > 1:
            previous_move = self.player_move_history[-2]
        
        # Log the round data with enhanced statistics
        round_data = {
            "round_number": self.rounds_played,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_choice": user_choice,
            "computer_choice": computer_choice,
            "winner": winner,
            "ai_mode": self.ai_mode,
            "ai_difficulty": self.ai_difficulty,
            "user_score": self.score["user"],
            "computer_score": self.score["computer"],
            "decision_time": decision_time,
            "current_streak": current_streak,
            "streak_type": streak_type,
            "previous_user_move": previous_move,
            "detection_duration": self.detection_duration,
            "match_progress": f"{self.rounds_played}/{self.max_rounds}"
        }
        
        # Add to in-memory history and log to file
        self.game_history.append(round_data)
        self.log_round(round_data)
        
        return result
            
    def play_sound(self, sound_type):
        """Play a sound effect based on the sound type"""
        try:
            if sound_type in self.sound_effects:
                frequency = self.sound_effects[sound_type]
                duration = 300  # milliseconds
                winsound.Beep(frequency, duration)
        except:
            # Silently fail if sound can't be played
            pass
            
    def reset_game(self):
        """Reset the game state for a new game"""
        self.score = {"user": 0, "computer": 0}
        self.rounds_played = 0
        self.game_state = "countdown"
        self.countdown = 3
        self.last_countdown_time = time.time()
        self.detected_gestures = []
        
        # Don't reset player move history - AI should continue learning
    
    def display_text(self, frame, text, position, font_scale=1, color=(255, 255, 255), thickness=2, center=False):
        """
        Display text on the frame with optional centering
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # If center is True, calculate the position to center the text
        if center:
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = position[1]
            position = (text_x, text_y)
        
        # Add a dark background behind the text for better readability
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        cv2.rectangle(frame, 
                     (position[0] - 5, position[1] - text_size[1] - 5), 
                     (position[0] + text_size[0] + 5, position[1] + 5), 
                     (0, 0, 0), -1)
        
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
    
    def create_game_ui(self, frame):
        """
        Create a consistent game UI with a title bar and footer
        """
        # Create a title bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (50, 50, 50), -1)
        self.display_text(frame, "ROCK PAPER SCISSORS", (0, 35), 1.2, self.title_color, 2, center=True)
        
        # Create a footer with more control options
        cv2.rectangle(frame, (0, frame.shape[0]-40), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
        footer_text = "Q: Quit | K: Keyboard | R: Restart | M: Menu | +/-: Adjust Difficulty"
        self.display_text(frame, footer_text, (0, frame.shape[0]-15), 0.7, (200, 200, 200), 2, center=True)
        
        # Display score and round information below the title bar
        score_bar_height = 30
        cv2.rectangle(frame, (0, 50), (frame.shape[1], 50 + score_bar_height), (30, 30, 30), -1)
        
        # Add animation to the score display when it changes
        if self.frame_count % self.animation_speed == 0:
            self.animation_frame = (self.animation_frame + 1) % 10
        self.frame_count += 1
        
        # Highlight the score that changed most recently with a pulsing effect
        user_color = self.score_color
        computer_color = self.score_color
        
        if self.result and "win" in self.result.lower() and "computer" not in self.result.lower():
            # User won, highlight user score
            highlight_intensity = 155 + 100 * abs(math.sin(self.animation_frame * 0.2))
            user_color = (0, highlight_intensity, 0)
        elif self.result and "computer" in self.result.lower():
            # Computer won, highlight computer score
            highlight_intensity = 155 + 100 * abs(math.sin(self.animation_frame * 0.2))
            computer_color = (0, 0, highlight_intensity)
        
        # Display round information
        round_text = f"ROUND: {self.rounds_played + 1}/{self.max_rounds}" if self.rounds_played < self.max_rounds else "FINAL RESULTS"
        self.display_text(frame, round_text, (10, 50 + score_bar_height - 10), 0.7, (150, 150, 255), 2)
        
        # Display scores with potential highlighting
        user_score_text = f"YOU: {self.score['user']}"
        comp_score_text = f"COMPUTER: {self.score['computer']}"
        
        # Calculate positions to space them out evenly
        frame_width = frame.shape[1]
        self.display_text(frame, user_score_text, (frame_width//2 - 100, 50 + score_bar_height - 10), 0.7, user_color, 2)
        self.display_text(frame, comp_score_text, (frame_width//2 + 20, 50 + score_bar_height - 10), 0.7, computer_color, 2)
    
    def draw_menu(self, frame, game_area_center_y):
        """Draw the game menu with options"""
        menu_bg_height = 400
        menu_bg_width = 500
        menu_x = (frame.shape[1] - menu_bg_width) // 2
        menu_y = (frame.shape[0] - menu_bg_height) // 2
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (menu_x, menu_y), 
                     (menu_x + menu_bg_width, menu_y + menu_bg_height), 
                     (30, 30, 50), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw menu title
        self.display_text(frame, "GAME MENU", 
                         (0, menu_y + 40), 1.5, self.menu_color, 3, center=True)
        
        # Draw menu options
        options = [
            "1: Start New Game",
            "2: Change Difficulty",
            "3: Toggle Sound",
            "4: Help",
            "5: AI Strategy Settings",
            "6: View Game Stats",
            "Q: Quit Game"
        ]
        
        for i, option in enumerate(options):
            y_pos = menu_y + 100 + i * 40
            self.display_text(frame, option, 
                             (menu_x + 50, y_pos), 1, (255, 255, 255), 2)
        
        # Display current AI mode
        ai_mode_text = f"Current AI: {self.ai_mode.capitalize()} (Level: {int(self.ai_difficulty * 100)}%)"
        self.display_text(frame, ai_mode_text, 
                         (menu_x + 50, menu_y + 380), 0.8, (150, 255, 150), 2)
            
        # Draw animated selection indicator
        indicator_pos = menu_y + 100 + (self.animation_frame % len(options)) * 40
        cv2.circle(frame, (menu_x + 30, indicator_pos - 5), 8, 
                  (0, 255, 255), -1)
                  
    def draw_ai_settings(self, frame, game_area_center_y):
        """Draw the AI strategy settings menu"""
        menu_bg_height = 350
        menu_bg_width = 500
        menu_x = (frame.shape[1] - menu_bg_width) // 2
        menu_y = (frame.shape[0] - menu_bg_height) // 2
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (menu_x, menu_y), 
                     (menu_x + menu_bg_width, menu_y + menu_bg_height), 
                     (30, 50, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw menu title
        self.display_text(frame, "AI STRATEGY SETTINGS", 
                         (0, menu_y + 40), 1.5, (100, 255, 100), 3, center=True)
        
        # Draw AI mode options
        ai_modes = [
            "1: Random (AI plays randomly)",
            "2: Counter (AI counters your most frequent move)",
            "3: Pattern (AI learns your patterns)",
            "4: Adaptive (AI combines strategies dynamically)",
        ]
        
        for i, mode in enumerate(ai_modes):
            y_pos = menu_y + 100 + i * 40
            # Highlight current mode
            color = (255, 255, 0) if self.ai_mode == mode.split(":")[0].lower().split()[0] else (255, 255, 255)
            self.display_text(frame, mode, (menu_x + 50, y_pos), 1, color, 2)
        
        # Draw difficulty slider
        slider_y = menu_y + 230
        slider_width = 300
        slider_height = 20
        slider_x = menu_x + (menu_bg_width - slider_width) // 2
        
        # Draw slider background
        cv2.rectangle(frame, (slider_x, slider_y), 
                     (slider_x + slider_width, slider_y + slider_height), 
                     (100, 100, 100), -1)
        
        # Draw slider position
        position = int(self.ai_difficulty * slider_width)
        cv2.rectangle(frame, (slider_x, slider_y), 
                     (slider_x + position, slider_y + slider_height), 
                     (0, 255, 0), -1)
        
        # Draw slider labels
        self.display_text(frame, "AI Difficulty:", 
                         (slider_x, slider_y - 10), 0.8, (255, 255, 255), 2)
        self.display_text(frame, "Easy", 
                         (slider_x, slider_y + 40), 0.7, (255, 255, 255), 1)
        self.display_text(frame, "Hard", 
                         (slider_x + slider_width - 40, slider_y + 40), 0.7, (255, 255, 255), 1)
        
        # Display current settings
        current_settings = f"Current: {self.ai_mode.capitalize()} mode, {int(self.ai_difficulty * 100)}% difficulty"
        self.display_text(frame, current_settings, 
                         (0, slider_y + 80), 1, (255, 255, 0), 2, center=True)
        
        # Instructions
        self.display_text(frame, "Press 1-3 to change mode, +/- to adjust difficulty, B to go back", 
                         (0, menu_y + 320), 0.7, (200, 200, 200), 1, center=True)
                         
    def draw_stats(self, frame, game_area_center_y):
        """Draw game statistics from the log file"""
        stats_bg_height = 500
        stats_bg_width = 700
        stats_x = (frame.shape[1] - stats_bg_width) // 2
        stats_y = (frame.shape[0] - stats_bg_height) // 2
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (stats_x, stats_y), 
                     (stats_x + stats_bg_width, stats_y + stats_bg_height), 
                     (50, 30, 50), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw title
        self.display_text(frame, "GAME STATISTICS", 
                         (0, stats_y + 40), 1.5, (200, 150, 255), 3, center=True)
        
        try:
            # Use in-memory history if available, otherwise read from file
            if self.game_history:
                rounds = self.game_history
            else:
                # Read log data from file
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
                    rounds = log_data.get("rounds", [])
            
            if not rounds:
                self.display_text(frame, "No game data available yet", 
                                 (0, stats_y + 200), 1, (255, 255, 255), 2, center=True)
            else:
                # Calculate basic statistics
                total_rounds = len(rounds)
                user_wins = sum(1 for r in rounds if r.get("winner") == "user")
                computer_wins = sum(1 for r in rounds if r.get("winner") == "computer")
                ties = sum(1 for r in rounds if r.get("winner") == "tie")
                
                # Calculate win rates
                user_win_rate = int(user_wins/total_rounds*100) if total_rounds > 0 else 0
                computer_win_rate = int(computer_wins/total_rounds*100) if total_rounds > 0 else 0
                tie_rate = int(ties/total_rounds*100) if total_rounds > 0 else 0
                
                # Count move frequencies
                user_moves = [r.get("user_choice") for r in rounds]
                move_counts = Counter(user_moves)
                rock_count = move_counts.get("rock", 0)
                paper_count = move_counts.get("paper", 0)
                scissors_count = move_counts.get("scissors", 0)
                
                # Calculate move percentages
                rock_pct = int(rock_count/total_rounds*100) if total_rounds > 0 else 0
                paper_pct = int(paper_count/total_rounds*100) if total_rounds > 0 else 0
                scissors_pct = int(scissors_count/total_rounds*100) if total_rounds > 0 else 0
                
                # Calculate streaks
                max_user_streak = max([r.get("current_streak", 0) for r in rounds if r.get("streak_type") == "user"], default=0)
                max_computer_streak = max([r.get("current_streak", 0) for r in rounds if r.get("streak_type") == "computer"], default=0)
                
                # Calculate move transitions (what move tends to follow what)
                transitions = {"rock": {"rock": 0, "paper": 0, "scissors": 0},
                              "paper": {"rock": 0, "paper": 0, "scissors": 0},
                              "scissors": {"rock": 0, "paper": 0, "scissors": 0}}
                
                for i in range(1, len(user_moves)):
                    prev_move = user_moves[i-1]
                    curr_move = user_moves[i]
                    transitions[prev_move][curr_move] += 1
                
                # Find most common transitions
                most_common_after = {}
                for move, follows in transitions.items():
                    if sum(follows.values()) > 0:
                        most_common = max(follows.items(), key=lambda x: x[1])
                        most_common_after[move] = most_common[0]
                
                # Calculate AI effectiveness
                ai_modes_used = Counter([r.get("ai_mode") for r in rounds])
                ai_mode_wins = {}
                for mode in ai_modes_used:
                    mode_rounds = [r for r in rounds if r.get("ai_mode") == mode]
                    mode_wins = sum(1 for r in mode_rounds if r.get("winner") == "computer")
                    ai_mode_wins[mode] = f"{mode_wins}/{len(mode_rounds)} ({int(mode_wins/len(mode_rounds)*100)}%)" if len(mode_rounds) > 0 else "N/A"
                
                # Display statistics in columns
                col1_x = stats_x + 50
                col2_x = stats_x + stats_bg_width // 2 + 20
                
                # Column 1: Basic Stats
                col1_stats = [
                    f"Total Rounds: {total_rounds}",
                    f"Your Wins: {user_wins} ({user_win_rate}%)",
                    f"Computer Wins: {computer_wins} ({computer_win_rate}%)",
                    f"Ties: {ties} ({tie_rate}%)",
                    "",
                    "Your Move Choices:",
                    f"Rock: {rock_count} ({rock_pct}%)",
                    f"Paper: {paper_count} ({paper_pct}%)",
                    f"Scissors: {scissors_count} ({scissors_pct}%)",
                    "",
                    "Longest Streaks:",
                    f"Your Streak: {max_user_streak}",
                    f"Computer Streak: {max_computer_streak}"
                ]
                
                # Column 2: Advanced Stats
                col2_stats = [
                    "Move Patterns:",
                ]
                
                # Add transition patterns
                for move, next_move in most_common_after.items():
                    col2_stats.append(f"After {move}: {next_move} ({int(transitions[move][next_move]/sum(transitions[move].values())*100)}%)")
                
                col2_stats.extend([
                    "",
                    "AI Performance:",
                ])
                
                # Add AI mode performance
                for mode, win_rate in ai_mode_wins.items():
                    col2_stats.append(f"{mode.capitalize()}: {win_rate}")
                
                # Display Column 1
                for i, stat in enumerate(col1_stats):
                    y_pos = stats_y + 100 + i * 25
                    self.display_text(frame, stat, 
                                     (col1_x, y_pos), 0.7, (255, 255, 255), 1)
                
                # Display Column 2
                for i, stat in enumerate(col2_stats):
                    y_pos = stats_y + 100 + i * 25
                    self.display_text(frame, stat, 
                                     (col2_x, y_pos), 0.7, (255, 255, 255), 1)
                
                # Draw a visual representation of move distribution
                pie_center_x = col1_x + 100
                pie_center_y = stats_y + 400
                pie_radius = 50
                
                # Draw pie chart segments
                start_angle = 0
                
                # Rock segment (red)
                rock_angle = 360 * (rock_count / total_rounds) if total_rounds > 0 else 0
                cv2.ellipse(frame, (pie_center_x, pie_center_y), (pie_radius, pie_radius), 
                           0, start_angle, start_angle + rock_angle, (0, 0, 255), -1)
                start_angle += rock_angle
                
                # Paper segment (green)
                paper_angle = 360 * (paper_count / total_rounds) if total_rounds > 0 else 0
                cv2.ellipse(frame, (pie_center_x, pie_center_y), (pie_radius, pie_radius), 
                           0, start_angle, start_angle + paper_angle, (0, 255, 0), -1)
                start_angle += paper_angle
                
                # Scissors segment (blue)
                scissors_angle = 360 * (scissors_count / total_rounds) if total_rounds > 0 else 0
                cv2.ellipse(frame, (pie_center_x, pie_center_y), (pie_radius, pie_radius), 
                           0, start_angle, start_angle + scissors_angle, (255, 0, 0), -1)
                
                # Draw pie chart legend
                legend_x = pie_center_x + pie_radius + 20
                self.display_text(frame, "Move Distribution:", 
                                 (legend_x, pie_center_y - 40), 0.7, (255, 255, 255), 1)
                cv2.rectangle(frame, (legend_x, pie_center_y - 30), (legend_x + 15, pie_center_y - 15), (0, 0, 255), -1)
                self.display_text(frame, "Rock", 
                                 (legend_x + 20, pie_center_y - 20), 0.6, (255, 255, 255), 1)
                cv2.rectangle(frame, (legend_x, pie_center_y), (legend_x + 15, pie_center_y + 15), (0, 255, 0), -1)
                self.display_text(frame, "Paper", 
                                 (legend_x + 20, pie_center_y + 10), 0.6, (255, 255, 255), 1)
                cv2.rectangle(frame, (legend_x, pie_center_y + 30), (legend_x + 15, pie_center_y + 45), (255, 0, 0), -1)
                self.display_text(frame, "Scissors", 
                                 (legend_x + 20, pie_center_y + 40), 0.6, (255, 255, 255), 1)
                
        except Exception as e:
            self.display_text(frame, f"Error loading statistics: {e}", 
                             (0, stats_y + 200), 0.8, (255, 100, 100), 2, center=True)
        
        # Instructions and export option
        self.display_text(frame, "Press B to go back, E to export data", 
                         (0, stats_y + 470), 0.8, (200, 200, 200), 1, center=True)
    
    def show_notification(self, frame):
        """Display a temporary notification message"""
        if self.display_message and time.time() - self.display_message_time < self.display_message_duration:
            # Calculate remaining time
            remaining = self.display_message_duration - (time.time() - self.display_message_time)
            alpha = min(1.0, remaining)  # Fade out effect
            
            # Create notification background
            notification_height = 40
            notification_y = 50
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, notification_y), 
                         (frame.shape[1], notification_y + notification_height), 
                         (50, 50, 100), -1)
            cv2.addWeighted(overlay, 0.7 * alpha, frame, 1 - 0.7 * alpha, 0, frame)
            
            # Display message
            self.display_text(frame, self.display_message, 
                             (0, notification_y + 30), 0.8, 
                             (255, 255, 255), 2, center=True)
    
    def draw_help_screen(self, frame, game_area_center_y):
        """Draw the help screen with instructions"""
        help_bg_height = 400
        help_bg_width = 600
        help_x = (frame.shape[1] - help_bg_width) // 2
        help_y = (frame.shape[0] - help_bg_height) // 2
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (help_x, help_y), 
                     (help_x + help_bg_width, help_y + help_bg_height), 
                     (30, 50, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw help title
        self.display_text(frame, "HOW TO PLAY", 
                         (0, help_y + 40), 1.5, (100, 255, 100), 3, center=True)
        
        # Draw instructions
        instructions = [
            "1. Show your hand in the detection area",
            "2. Make one of these gestures:",
            "   - Rock: Make a fist (close all fingers)",
            "   - Paper: Open palm with all fingers extended",
            "   - Scissors: Extend only index and middle fingers",
            "3. Hold your gesture until the timer ends",
            "4. The computer will randomly choose its move",
            "5. Winner is determined by standard rules:",
            "   - Rock beats Scissors",
            "   - Scissors beats Paper",
            "   - Paper beats Rock",
            "",
            "Press any key to return to the game"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = help_y + 80 + i * 25
            self.display_text(frame, instruction, 
                             (help_x + 20, y_pos), 0.7, (220, 220, 220), 1)
    
    def run(self):
        # Set the ROI for hand detection based on frame size
        roi_width = self.width // 2
        roi_height = self.height // 2
        roi_left = (self.width - roi_width) // 2
        roi_top = (self.height - roi_height) // 2 + 50  # Offset for title and score bars
        self.hand_detector.set_roi(roi_top, roi_top + roi_height, roi_left, roi_left + roi_width)
        
        use_keyboard = False  # Flag to toggle between keyboard and gesture detection
        show_help = False     # Flag to show help screen
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Flip the frame horizontally for a more natural view
            frame = cv2.flip(frame, 1)
            
            # Apply the game UI
            self.create_game_ui(frame)
            
            # Show any active notifications
            self.show_notification(frame)
            
            # Calculate the center of the game area (accounting for title and score bars)
            title_height = 50
            score_bar_height = 30
            footer_height = 40
            game_area_center_y = (frame.shape[0] - (title_height + score_bar_height + footer_height)) // 2 + title_height + score_bar_height
            
            # Check if game is over (best of N rounds)
            if self.rounds_played >= self.max_rounds and self.game_state != "menu":
                # Game is over, show final results
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                self.display_text(frame, "GAME OVER", 
                                 (0, game_area_center_y - 100), 2, (255, 255, 255), 3, center=True)
                
                if self.score["user"] > self.score["computer"]:
                    result_text = "YOU WIN THE MATCH!"
                    result_color = (0, 255, 0)
                elif self.score["user"] < self.score["computer"]:
                    result_text = "COMPUTER WINS THE MATCH!"
                    result_color = (0, 0, 255)
                else:
                    result_text = "THE MATCH IS A TIE!"
                    result_color = (255, 255, 0)
                
                self.display_text(frame, result_text, 
                                 (0, game_area_center_y), 1.5, result_color, 3, center=True)
                
                self.display_text(frame, f"Final Score: You {self.score['user']} - Computer {self.score['computer']}", 
                                 (0, game_area_center_y + 70), 1, (200, 200, 200), 2, center=True)
                
                self.display_text(frame, "Press 'R' to play again or 'Q' to quit", 
                                 (0, game_area_center_y + 140), 1, (150, 150, 255), 2, center=True)
                
                # Wait for user input
                key = cv2.waitKey(1)
                if key == ord('r'):
                    self.reset_game()
                elif key == ord('q'):
                    break
                
                # Display the frame and continue
                cv2.imshow('Rock Paper Scissors Game', frame)
                continue
            
            # Game state handling
            if self.game_state == "menu":
                self.draw_menu(frame, game_area_center_y)
            elif self.game_state == "ai_settings":
                self.draw_ai_settings(frame, game_area_center_y)
            elif self.game_state == "stats":
                self.draw_stats(frame, game_area_center_y)
            elif show_help:
                self.draw_help_screen(frame, game_area_center_y)
            elif self.game_state == "countdown":
                current_time = time.time()
                if current_time - self.last_countdown_time >= 1:
                    self.countdown -= 1
                    self.last_countdown_time = current_time
                    # Play countdown sound
                    self.play_sound("countdown")
                
                # Display countdown in the center with a large font
                self.display_text(frame, f"GET READY!", 
                                 (0, game_area_center_y - 50), 1.5, self.countdown_color, 3, center=True)
                self.display_text(frame, f"{self.countdown}", 
                                 (0, game_area_center_y + 30), 3, self.countdown_color, 5, center=True)
                
                if self.countdown <= 0:
                    self.game_state = "detection"
                    self.computer_choice = self.get_computer_choice()
                    self.detection_start_time = time.time()
                    self.detected_gestures = []
                    self.no_hand_count = 0
                    self.detection_feedback = ""
            
            elif self.game_state == "detection":
                # Process hand detection if not using keyboard
                if not use_keyboard:
                    # Detect hand and get gesture
                    frame, hand_contour = self.hand_detector.detect_hands(frame)
                    detected_gesture = self.hand_detector.get_gesture(hand_contour)
                    
                    # If a gesture is detected, add it to the list and reset no_hand_count
                    if detected_gesture:
                        self.detected_gestures.append(detected_gesture)
                        self.no_hand_count = 0
                        
                        # Draw the detected gesture info
                        self.hand_detector.draw_gesture_info(frame, detected_gesture)
                        
                        # Update feedback based on detected gesture
                        if len(self.detected_gestures) > 2:
                            # Count recent gestures
                            recent_gestures = self.detected_gestures[-3:]
                            if all(g == recent_gestures[0] for g in recent_gestures):
                                self.detection_feedback = f"Good! Keep showing {detected_gesture.upper()}"
                                self.feedback_color = (0, 255, 0)
                            else:
                                self.detection_feedback = "Try to keep your gesture stable"
                                self.feedback_color = (0, 200, 255)
                    else:
                        # No hand detected
                        self.no_hand_count += 1
                        if self.no_hand_count > self.max_no_hand_count:
                            self.detection_feedback = "No hand detected! Place your hand in the box."
                            self.feedback_color = (0, 0, 255)
                    
                    # Calculate remaining detection time
                    elapsed_time = time.time() - self.detection_start_time
                    remaining_time = max(0, self.detection_duration - elapsed_time)
                    
                    # Display detection progress
                    progress_width = int((elapsed_time / self.detection_duration) * (self.width - 100))
                    cv2.rectangle(frame, (50, game_area_center_y + 100), 
                                 (50 + progress_width, game_area_center_y + 110), 
                                 (0, 255, 0), -1)
                    cv2.rectangle(frame, (50, game_area_center_y + 100), 
                                 (self.width - 50, game_area_center_y + 110), 
                                 (255, 255, 255), 2)
                    
                    # Display instructions
                    self.display_text(frame, "SHOW YOUR HAND GESTURE!", 
                                     (0, game_area_center_y - 60), 1.2, self.instruction_color, 2, center=True)
                    
                    # Display detection time
                    self.display_text(frame, f"Detecting: {remaining_time:.1f}s", 
                                     (0, game_area_center_y), 0.8, self.instruction_color, 2, center=True)
                    
                    # Display feedback
                    if self.detection_feedback:
                        self.display_text(frame, self.detection_feedback, 
                                         (0, game_area_center_y + 40), 0.8, self.feedback_color, 2, center=True)
                    
                    # Computer's choice is hidden until user makes a selection
                    self.display_text(frame, "Computer is waiting for your move...", 
                                     (0, game_area_center_y + 70), 0.9, (100, 100, 255), 2, center=True)
                    
                    # Check if detection time is up
                    if elapsed_time >= self.detection_duration and self.detected_gestures:
                        # Count occurrences of each gesture
                        gesture_counts = {}
                        for gesture in self.detected_gestures:
                            if gesture in gesture_counts:
                                gesture_counts[gesture] += 1
                            else:
                                gesture_counts[gesture] = 1
                        
                        # Find the most common gesture
                        most_common_gesture = max(gesture_counts, key=gesture_counts.get)
                        
                        # Check if the most common gesture appears enough times
                        if gesture_counts[most_common_gesture] >= self.gesture_confidence_threshold:
                            self.user_choice = most_common_gesture
                            self.result = self.determine_winner(self.user_choice, self.computer_choice)
                            self.game_state = "result"
                            self.result_display_time = time.time()
                        else:
                            # Not confident enough, restart detection with feedback
                            self.detection_start_time = time.time()
                            self.detected_gestures = []
                            self.detection_feedback = "Unclear gesture. Please try again."
                            self.feedback_color = (0, 0, 255)
                            self.play_sound("error")
                else:
                    # Using keyboard input
                    self.display_text(frame, "KEYBOARD MODE ACTIVE", 
                                     (0, game_area_center_y - 60), 1.2, (255, 0, 0), 2, center=True)
                    
                    # Display key controls with better spacing
                    self.display_text(frame, "Press: R (Rock) | P (Paper) | S (Scissors)", 
                                     (0, game_area_center_y), 0.8, self.instruction_color, 2, center=True)
                    
                    # Computer's choice is hidden until user makes a selection
                    self.display_text(frame, "Computer is waiting for your move...", 
                                     (0, game_area_center_y + 60), 0.9, (100, 100, 255), 2, center=True)
                
            elif self.game_state == "result":
                if time.time() - self.result_display_time >= 3:  # Show result for 3 seconds
                    if self.rounds_played >= self.max_rounds:
                        # Game is over, next frame will show final results
                        pass
                    else:
                        # Continue to next round
                        self.game_state = "countdown"
                        self.countdown = 3
                        self.last_countdown_time = time.time()
                else:
                    # Create animated background for result
                    overlay = frame.copy()
                    bg_color = (0, 100, 0) if "win" in self.result.lower() and "computer" not in self.result.lower() else \
                              (100, 0, 0) if "computer" in self.result.lower() else (100, 100, 0)
                    cv2.rectangle(overlay, (0, game_area_center_y - 100), 
                                 (frame.shape[1], game_area_center_y + 100), bg_color, -1)
                    # Pulsing animation
                    alpha = 0.5 + 0.3 * abs(math.sin(self.animation_frame * 0.2))
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    
                    # Display the result in the center with better formatting
                    self.display_text(frame, f"YOU CHOSE: {self.user_choice.upper()}", 
                                     (0, game_area_center_y - 80), 1, (200, 255, 255), 2, center=True)
                    self.display_text(frame, f"COMPUTER CHOSE: {self.computer_choice.upper()}", 
                                     (0, game_area_center_y - 30), 1, (200, 255, 255), 2, center=True)
                    
                    # Display the result with a larger font and highlight
                    result_color = (0, 255, 0) if "win" in self.result.lower() and "computer" not in self.result.lower() else \
                                  (0, 0, 255) if "computer" in self.result.lower() else (255, 255, 0)
                    self.display_text(frame, self.result.upper(), 
                                     (0, game_area_center_y + 40), 1.5, result_color, 3, center=True)
            
            # Display the frame
            cv2.imshow('Rock Paper Scissors Game', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('k'):
                # Toggle between keyboard and gesture detection
                use_keyboard = not use_keyboard
                if use_keyboard:
                    print("Switched to keyboard mode")
                else:
                    print("Switched to gesture detection mode")
                    self.detection_start_time = time.time()
                    self.detected_gestures = []
            elif key == ord('r'):
                # Restart the game
                self.reset_game()
                show_help = False
            elif key == ord('m'):
                # Show/hide menu
                if self.game_state != "menu":
                    self.game_state = "menu"
                else:
                    self.game_state = "countdown"
                    self.countdown = 3
                    self.last_countdown_time = time.time()
            elif key == ord('h') or key == ord('4'):
                # Toggle help screen
                show_help = not show_help
            elif key == ord('+') or key == ord('='):
                # Increase difficulty (shorter detection time)
                self.detection_duration = max(1, self.detection_duration - 0.5)
                print(f"Detection time: {self.detection_duration}s")
            elif key == ord('-') or key == ord('_'):
                # Decrease difficulty (longer detection time)
                self.detection_duration = min(5, self.detection_duration + 0.5)
                print(f"Detection time: {self.detection_duration}s")
            elif self.game_state == "detection" and use_keyboard:
                if key == ord('r'):
                    self.user_choice = 'rock'
                    self.result = self.determine_winner(self.user_choice, self.computer_choice)
                    self.game_state = "result"
                    self.result_display_time = time.time()
                elif key == ord('p'):
                    self.user_choice = 'paper'
                    self.result = self.determine_winner(self.user_choice, self.computer_choice)
                    self.game_state = "result"
                    self.result_display_time = time.time()
                elif key == ord('s'):
                    self.user_choice = 'scissors'
                    self.result = self.determine_winner(self.user_choice, self.computer_choice)
                    self.game_state = "result"
                    self.result_display_time = time.time()
            elif show_help and key != -1:
                # Any key exits help screen
                show_help = False
            elif self.game_state == "menu":
                if key == ord('1'):
                    # Start new game
                    self.reset_game()
                elif key == ord('2'):
                    # Change difficulty
                    self.detection_duration = 3.0  # Reset to default
                    self.game_state = "countdown"
                elif key == ord('3'):
                    # Toggle sound (just a placeholder, we'd need more state for this)
                    pass
                elif key == ord('4'):
                    # Show help
                    show_help = True
                elif key == ord('5'):
                    # AI Strategy Settings
                    self.game_state = "ai_settings"
                elif key == ord('6'):
                    # View Game Stats
                    self.game_state = "stats"
            elif self.game_state == "ai_settings":
                if key == ord('1'):
                    # Random AI
                    self.ai_mode = "random"
                elif key == ord('2'):
                    # Counter AI
                    self.ai_mode = "counter"
                elif key == ord('3'):
                    # Pattern AI
                    self.ai_mode = "pattern"
                elif key == ord('4'):
                    # Adaptive AI
                    self.ai_mode = "adaptive"
                elif key == ord('+') or key == ord('='):
                    # Increase AI difficulty
                    self.ai_difficulty = min(1.0, self.ai_difficulty + 0.1)
                elif key == ord('-') or key == ord('_'):
                    # Decrease AI difficulty
                    self.ai_difficulty = max(0.0, self.ai_difficulty - 0.1)
                elif key == ord('b'):
                    # Back to menu
                    self.game_state = "menu"
            elif self.game_state == "stats":
                if key == ord('b'):
                    # Back to menu
                    self.game_state = "menu"
                elif key == ord('e'):
                    # Export data and show confirmation
                    csv_file = self.export_to_csv()
                    if csv_file:
                        # Show export confirmation message
                        self.display_message = f"Data exported to {os.path.basename(csv_file)}"
                        self.display_message_time = time.time()
                        self.display_message_duration = 3  # seconds to show message
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = RockPaperScissorsGame()
    game.run()