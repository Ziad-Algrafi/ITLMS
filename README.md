# ITLMS: An Intelligent Traffic Light Management System Using Edge Computing and Deep Learning

## Abstract

This repository contains the implementation of an Intelligent Traffic Light Management System (ITLMS) as described in our paper. The system uses computer vision and deep learning to dynamically control traffic lights based on real-time analysis of traffic conditions, with priority given to emergency vehicles.

## Key Features

- Real-time vehicle detection and classification using YOLOv9
- Dynamic traffic light control based on:
  - Emergency vehicle presence (ambulance, firefighter, police)
  - Traffic density calculations
- Graphical user interface for monitoring traffic conditions
- Edge computing implementation for efficient processing

## Requirements
- Python 3.10+
- OpenCV
- Pillow
- Ultralytics
- Tkinter

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Ziad-Algrafi/ITLMS.git
   cd traffic-management-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv9 model weights (best.pt) and place it in the project directory

## Configuration
Before running the application, you need to configure the video paths in `main.py`:
```python
video_paths_and_roads = {
    "Road1": {"video_path": "/path/to/video1.mp4", "road": 1},
    "Road2": {"video_path": "/path/to/video2.mp4", "road": 2},
    "Road3": {"video_path": "/path/to/video3.mp4", "road": 3},
    "Road4": {"video_path": "/path/to/video4.mp4", "road": 4},
}
```

## Usage
Run the application:
```bash
python main.py
```

The GUI will display:
- Live video feeds from each road
- Vehicle counts (ambulance, firefighter, police, traffic jam, cars)
- Traffic light status (Open/Closed)

## How It Works
1. The system processes video feeds from multiple roads
2. YOLOv9 detects and classifies vehicles
3. The traffic light system prioritizes:
   - Emergency vehicles (ambulance > firefighter > police)
   - Roads with highest traffic density when no emergencies exist
4. Traffic lights stay open for a minimum of 10 seconds before switching


## License

Apache-2.0 license
