# Real-time Face Detection and Gaze Tracking

This is a Flask application that performs real-time face detection and gaze tracking using OpenCV, dlib, and Flask. It streams the webcam feed to a web interface while detecting faces and displaying their landmarks.

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- Flask (`pip install flask`)
- dlib (requires CMake and Visual Studio C++ tools)

## Installing dlib

To install dlib, you'll need to have CMake and Visual Studio C++ tools installed on your system.

1. Install CMake from [here](https://cmake.org/download/).
2. Install Visual Studio C++ tools from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

Once you have CMake and Visual Studio C++ tools installed, you can install dlib using pip:

```bash
pip install dlib
```

## Running the Application

1. Clone this repository to your local machine.
2. Navigate to the directory where you cloned the repository.
3. Run the Flask application by executing the following command:

```bash
python app.py
```

