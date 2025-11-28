# AI/ML Football Analysis System

![Image](https://github.com/user-attachments/assets/bd709b75-2705-4d52-806f-c4798276d3ef)
*(Gameplay overview)*

## ⚽ Project Overview
**Computer Vision Scout** is an advanced AI-powered system designed to analyze football match footage. It leverages state-of-the-art computer vision models (YOLOv8) and machine learning techniques to detect players, track their movements, identify teams, and provide actionable insights like speed estimation and possession statistics.

This project demonstrates the power of unstructured data analysis in sports analytics, transforming raw video into structured data and broadcast-quality visualizations.

## 🚀 Key Features
- **Object Detection**: Real-time detection of players, referees, and the ball using **YOLOv8**.
- **Player Tracking**: Robust multi-object tracking using **ByteTrack** to follow players across frames.
- **Team Identification**: Automatic team separation based on jersey colors using **K-Means Clustering**.
- **Speed Estimation**: Calculates player speed (km/h) based on pixel movement and perspective transformation.
- **Possession Stats**: Real-time ball control percentage for each team.
- **Visualizations**:
    - Broadcast-style ellipses under players.
    - Clean, readable labels with player IDs and speed.
    - **Heatmaps** showing player movement density.
- **Player Recognition (Experimental)**: OCR-based jersey number reading to identify specific players.

## 🛠️ Tech Stack
- **Python 3.8+**
- **YOLOv8** (Ultralytics) - Object Detection
- **OpenCV** - Video Processing & Visualization
- **Supervision** - Tracking & Annotation Utilities
- **Scikit-Learn** - K-Means Clustering
- **EasyOCR** - Optical Character Recognition
- **NumPy & Pandas** - Data Manipulation

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/fauzan111/AI-ML_Football_Analysis_system_with_YOLO_OpenCV_and_Python.git
    cd AI-ML_Football_Analysis_system_with_YOLO_OpenCV_and_Python
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Input Video**
    - Place your football match video file in the project directory and name it `input.mp4`.

## ▶️ Usage

### 1. Run the Analysis
Execute the main script to process the video:
```bash
python main.py
```
This will:
- Process `input.mp4` frame by frame.
- Display progress in the terminal.
- Generate `output_video.avi` with all annotations.
- Save tracking data to `tracks.pkl`.

### 2. Generate Heatmap
After the analysis is complete, generate a movement heatmap:
```bash
python heatmap_generator.py
```
This will save `heatmap_output.png`.

## ⚙️ Configuration
- **Player Database**: Edit `player_database.py` to map jersey numbers to real player names for your specific video.
  ```python
  PLAYER_DATABASE = {
      'Team 1': {10: 'Messi', ...},
      'Team 2': {7: 'Ronaldo', ...}
  }
  Link for ouput video- https://drive.google.com/file/d/1p7mkqZYVr3MhHoavhOL6PgG8hQusy0I7/view?usp=drive_link
  ```
- **Models**: The system uses `yolov8m.pt` by default. You can change this in `main.py` for speed (n/s) or accuracy (l/x).

## 📊 How It Works
1.  **Detection**: YOLOv8 scans each frame to find people and the ball.
2.  **Tracking**: ByteTrack assigns unique IDs to each detected person to track them over time.
3.  **Team Assignment**: The system crops the player's jersey, extracts the dominant color, and clusters them into two teams.
4.  **Speed & Distance**: By assuming a standard player height, the system converts pixel distance to meters to estimate speed.
5.  **Visualization**: Custom annotators draw the ellipses, labels, and overlays on the video frames.

## 📝 License
This project is open-source and available under the MIT License.
