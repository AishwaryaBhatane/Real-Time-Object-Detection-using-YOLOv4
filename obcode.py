import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import os


class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Object Detection using YOLOv4")

        # ===== ABSOLUTE PATH FIX (CRITICAL) =====
        base_dir = os.path.dirname(os.path.abspath(__file__))

        cfg_path = os.path.join(base_dir, "yolov4.cfg")
        weights_path = os.path.join(base_dir, "yolov4.weights")
        names_path = os.path.join(base_dir, "coco.names")

        # ===== LOAD YOLO =====
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # ===== LOAD CLASS NAMES =====
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        self.output_layers = [
            layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()
        ]

        # ===== CAMERA (INTEGRATED CAMERA) =====
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print("ERROR: Camera not accessible")
            return

        self.label = Label(self.root)
        self.label.pack()

        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def detect_objects(self, frame):
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False
        )

        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]

                if confidence > 0.5:
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(cx - w / 2)
                    y = int(cy - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                conf = confidences[i]

                cv2.rectangle(frame, (x, y), (x + w, y + h), ( 255,0, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        return frame

    def update_frame(self):
        ret, frame = self.cap.read()

        if ret:
            frame = self.detect_objects(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.root.after(20, self.update_frame)

    def on_close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
