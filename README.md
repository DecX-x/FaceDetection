# Face Detection with OpenCV, TensorFlow and LabelMe
This repository provides code and instructions for training and deploying face detection models using OpenCV, TensorFlow, LabelMe and VGG16.

# Steps
It contains the following steps:

- Data Collection: Use OpenCV to collect images and video frames containing faces.
- Annotation: Use the LabelMe tool to annotate bounding boxes around faces in the dataset.
- Data Processing: Scripts to generate TFRecord files from the annotated datasets.
- Model Training: Train a TensorFlow model on the dataset using transfer learning with VGG16. Fine-tune a VGG16 model pre-trained on ImageNet for face detection.
- Evaluation: Assess model accuracy on a test dataset. Plot loss curves, precision, recall etc.
- Deployment: Optimize the VGG16 model for deployment. Convert to TensorFlow Lite or ONNX format.
- Inference: Run real-time face detection on video streams using the optimized VGG16 model in OpenCV.
The repository provides code and notebooks for each step listed above. Detailed guides explain how to train a VGG16 model for face detection, convert it for deployment and run inference using OpenCV.
