# Webcam Age Detection #

This project uses OpenCV and dlib to detect faces in real-time through a webcam feed and predict the age of the detected individuals. The age prediction is performed using a pre-trained deep learning model.

 ## Table of Contents ##
- Features
- Requirements
- Installation
- Usage


## Features ##  
- Real-time face detection using a webcam. 
- Predicts the age of detected faces.
- Displays the detected age on the video feed.
  
## Requirements ##
- Python 3.x
- OpenCV
- dlib
- NumPy
- 
You can install the required packages using pip:
```
pip install opencv-python dlib numpy
```

## Installation
1. Clone the repository:

```
git clone https://github.com/jitendra205saini/webcam_gender_finder_python.git
cd webcam_gender_finder_python
```
2. Download the pre-trained age model files and place them in the specified directories:

  - Age Model:
     - age_deploy.prototxt
     - age_net.caffemodel
    Ensure the paths in the code match the locations of these files.

## Usage ##
Run the script from the command line:

```
python your_script_name.py
```
## Example ##
```
python your_script_name.py
```
The webcam will open, and as it detects faces, it will predict and display the estimated age on the video feed.

## Key Code Features ##
- Face Detection: Uses dlib's frontal face detector to identify faces in the video frame.
- Age Prediction: Uses a pre-trained age estimation model to predict the age of detected faces.
- Error Handling: Includes error handling for model loading and face detection.
