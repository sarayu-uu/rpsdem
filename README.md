# CV-AI Showdown: Rock-Paper-Scissors

OpenCV | AI | Gesture | Real-time | Python

## Features

-   **Real-Time Gesture Recognition**: Uses your webcam to detect and classify hand gestures (Rock, Paper, Scissors) in real-time.
-   **Adaptive Hand Detection**: Implements skin color segmentation using the HSV color space, with an adaptive algorithm that cycles through different color ranges to adjust for varying lighting conditions.
-   **Advanced Game AI**: The computer opponent is powered by a multi-strategy AI that you can configure:
    -   **Random Mode**: The AI chooses its move randomly.
    -   **Counter Mode**: The AI analyzes your move history and counters your most frequent choice.
    -   **Pattern Mode**: The AI attempts to recognize and predict your move patterns.
    -   **Adaptive Mode**: The AI dynamically combines strategies based on its win/loss record against you, creating a challenging and unpredictable opponent.
-   **Dynamic UI with OpenCV**: The entire user interface is built from scratch using OpenCV, featuring:
    -   An interactive game menu to start new games, change settings, and view stats.
    -   Animated score tracking and visual feedback for game events.
    -   A real-time feedback system for gesture detection.
-   **In-Depth Statistics and Analytics**:
    -   All game rounds are logged to a JSON file for session persistence.
    -   A detailed statistics dashboard visualizes player and AI performance, including win rates, move distributions (with a pie chart), and AI strategy effectiveness.
    -   Option to export all game data to a CSV file for further analysis.
-   **Interactive Controls & Sound**:
    -   Full keyboard controls to navigate menus, change settings, and play the game.
    -   Sound effects (on Windows) for game events like countdowns, wins, and losses.

## Technical Deep Dive

### Hand Gesture Recognition

The core of the gesture recognition system is based on computer vision techniques applied to the webcam feed:

1.  **Region of Interest (ROI)**: The system focuses on a specific area of the screen to minimize background noise.
2.  **Skin Color Segmentation**: The frame is converted to the HSV color space to isolate pixels that fall within a predefined range for human skin tones. The system can adaptively switch between different ranges to handle various lighting conditions.
3.  **Contour Analysis**: After creating a binary mask of the detected skin, the system finds the largest contour, which is assumed to be the hand.
4.  **Convexity Defects**: The system analyzes the convex hull of the hand contour to find defects (i.e., the valleys between fingers). The number of defects is used to estimate the number of extended fingers.
5.  **Gesture Classification**: Based on the number of extended fingers, along with other metrics like solidity and circularity, the system classifies the gesture as Rock, Paper, or Scissors.

### Adaptive AI

The game's AI is designed to be more than just a random opponent. It learns from your gameplay to provide a more engaging challenge.

-   **Move History**: The AI keeps a history of your moves to identify patterns.
-   **Pattern Recognition**: The AI looks for repeating sequences, alternating patterns, and other common player behaviors.
-   **Counter Strategy**: The AI calculates your most frequent moves (with a recency bias) and chooses the move that beats it.
-   **Adaptive Logic**: In the most advanced mode, the AI uses its performance to decide which strategy to employ. If it's on a winning streak, it might stick with its current strategy. If it's losing, it will switch things up to try and break your pattern.

## Requirements

-   Python 3.x
-   OpenCV (`opencv-python`)
-   NumPy

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sarayu-uu/rpsdem.git
    cd rpsdem
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   **Windows:** `.\venv\Scripts\activate`
    -   **macOS/Linux:** `source venv/bin/activate`

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Play

1.  **Run the game:**
    ```bash
    python rock_paper_scissors.py
    ```
2.  The game will start in **countdown mode**.
3.  After the countdown, show your hand gesture (Rock, Paper, or Scissors) in the designated ROI on the screen.
4.  Hold your gesture steady until the detection timer completes.
5.  The computer will make its move, and the winner will be announced.
6.  The game continues for a best-of-5 match. You can restart or quit at any time.

## Controls

-   **Q**: Quit the game
-   **R**: Restart the game
-   **M**: Open/Close the game menu
-   **K**: Toggle between gesture detection and keyboard mode
-   **H**: Show the help screen
-   **+/-**: Adjust AI difficulty (in AI settings) or detection time

### Keyboard Mode Controls

When keyboard mode is active:

-   **R**: Choose Rock
-   **P**: Choose Paper
-   **S**: Choose Scissors
