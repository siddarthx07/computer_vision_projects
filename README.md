# 🧠 Computer Vision Projects

Welcome to the **Computer Vision Projects** repository! This repo contains a collection of practical and research-driven computer vision implementations using Python, OpenCV, and deep learning frameworks. These projects explore fundamental and advanced techniques across a variety of real-world tasks like object detection, image classification, and more.

---

## 📁 Repository Structure

```bash
computer_vision_projects/
├── basic_image_processing/
│   ├── edge_detection.py
│   └── color_filters.py
├── face_detection/
│   ├── haar_cascade.py
│   └── dnn_face_detector.py
├── object_detection/
│   ├── yolo_detection.py
│   └── utils/
├── image_classification/
│   ├── cnn_model.py
│   └── dataset_loader.py
├── license_plate_detection/
│   ├── plate_extractor.py
│   └── ocr_reader.py
└── README.md

Featured Projects
🔹 Basic Image Processing
Color space conversions (RGB ↔ HSV, Grayscale)

Edge detection (Sobel, Canny)

🔹 Face Detection
Real-time face detection using Haar Cascades

Deep Learning-based face detection (OpenCV DNN module)

🔹 Object Detection
YOLO-based object detection with bounding boxes

Integration with OpenCV for live video feed detection

🔹 Image Classification
Custom CNN model trained on sample datasets

Dataset preprocessing and augmentation support

🔹 License Plate Detection
Vehicle license plate localization using contour methods

OCR-based character recognition with Tesseract

⚙️ Installation
1. Clone the Repository

git clone https://github.com/siddarthx07/computer_vision_projects.git
cd computer_vision_projects
2. Create Virtual Environment (optional)


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies

pip install -r requirements.txt
If requirements.txt is missing, install key packages manually:


pip install opencv-python numpy matplotlib imutils tensorflow keras pytesseract

Sample Outputs
<table> <tr> <td><img src="docs/sample_face_detection.png" alt="Face Detection" width="250"/></td> <td><img src="docs/sample_object_detection.png" alt="Object Detection" width="250"/></td> <td><img src="docs/sample_plate_detection.png" alt="License Plate Detection" width="250"/></td> </tr> </table>

💡 Use Cases
Learning and experimenting with core computer vision algorithms

Prototyping ideas for object detection or recognition systems

Educational resource for CV courses and tutorials

🛠️ Tools & Libraries Used
Python

OpenCV

TensorFlow / Keras

PyTesseract

NumPy, Matplotlib, Imutils

🤝 Contributing
Feel free to fork the repo, open issues, or submit pull requests. Suggestions for new project ideas or optimizations are always welcome!
