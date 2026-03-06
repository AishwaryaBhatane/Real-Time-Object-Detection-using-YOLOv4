# Real-Time Object Detection using YOLOv4 🎯

## 📌 Project Overview

This project implements **real-time object detection using YOLOv4 and OpenCV** with a **Tkinter-based GUI**.
The application captures video from the system camera and detects objects in real time using the **COCO dataset**.

The detected objects are displayed with **bounding boxes and confidence scores** directly on the video feed.

# 🚀 Features

* Real-time object detection
* YOLOv4 deep learning model
* OpenCV DNN module
* Tkinter GUI interface
* Bounding boxes with confidence scores
* Non-Maximum Suppression for accurate detection

# 🛠 Technologies Used

* Python
* OpenCV
* YOLOv4
* Tkinter
* Pillow (PIL)
* COCO Dataset

# 📂 Project Structure

```
Object-Detection-YOLOv4
│
├── object_detection_app.py
├── yolov4.cfg
├── yolov4.weights
├── coco.names
└── README.md
```

# ⚙️ How It Works

1. The application loads the **YOLOv4 configuration and weight files**.
2. COCO class names are loaded from `coco.names`.
3. The system camera captures frames continuously.
4. Each frame is processed using the **YOLOv4 neural network**.
5. Objects are detected and displayed with:

   * Bounding boxes
   * Class labels
   * Confidence scores


# ▶️ How to Run

## Step 1: Install Dependencies

```bash
pip install opencv-python pillow
```
## Step 2: Download YOLO Files

Download the following files and place them in the same folder:

* yolov4.cfg
* yolov4.weights
* coco.names

You can download them from:

https://pjreddie.com/darknet/yolo/

## Step 3: Run the Application

bash
python object_detection_app.py


The camera window will open and start detecting objects in real time.

# 📊 Example Output

Detected objects will appear with:
Person 0.92
Dog 0.88
Bottle 0.75

# 📈 Skills Demonstrated

* Computer Vision
* Deep Learning Model Integration
* OpenCV DNN Module
* Real-time Video Processing
* GUI Development with Tkinter


# 🔮 Future Improvements

* GPU acceleration using CUDA
* Switch to YOLOv8 or YOLOv5
* Add screenshot capture feature
* Add object counting
* Deploy as a web application


