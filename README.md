# Rock Paper Scissors Game with Computer Vision

This project implements a Rock-Paper-Scissors game where you play against the computer using hand gestures captured by your webcam.

## Features

- Real-time webcam capture
- Hand gesture recognition (Rock, Paper, Scissors)
- Game logic to determine the winner
- Score tracking
- Visual feedback on the game state

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install the required packages:
   ```
   pip install opencv-python numpy
   ```

## Usage

Run the main game file:
```
python rock_paper_scissors.py
```

### How to Play

1. When the game starts, you'll see a countdown.
2. After the countdown, show your hand gesture (Rock, Paper, or Scissors).
3. The computer will randomly choose its gesture.
4. The winner will be determined and the score will be updated.
5. The game continues until you press 'q' to quit.

## Current Implementation

The current implementation uses keyboard input as a placeholder for hand gesture recognition:
- Press 'r' for Rock
- Press 'p' for Paper
- Press 's' for Scissors
- Press 'q' to quit the game

## Future Improvements

- Implement hand gesture recognition using computer vision techniques
- Add more visual effects and animations
- Improve the user interface
- Add difficulty levels for the computer player