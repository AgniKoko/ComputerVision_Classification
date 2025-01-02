# Multi-Class Classification and Object Detection in Computer Vision

## Project Description
This project involves implementing classification and object detection tasks in computer vision using Python. The tasks are divided into three parts, each focusing on a different methodology, including traditional machine learning, convolutional neural networks (CNNs), and object detection models. The project aims to explore Bag of Visual Words (BOVW), CNN architectures, and advanced object detection algorithms for traffic signs.

## Objectives
### **A. Multi-Class Classification with Bag of Visual Words**
- Implement multi-class classification using the Bag of Visual Words (BOVW) model.
- Test and evaluate two classifiers:
  - **k-NN** (k-nearest neighbors)
  - **SVM** (Support Vector Machine) with one-vs-all scheme.
- Measure accuracy using different vocabulary sizes and k-values for k-NN.

### **B. Multi-Class Classification with CNNs**
- Implement and compare two CNN architectures:
  1. A custom CNN designed from scratch for the classification task.
  2. A pre-trained CNN (e.g., ResNet50, MobileNet) adapted for the task using transfer learning.
- Evaluate performance based on classification accuracy.
- Analyze and justify design decisions, including data augmentation, batch size, and callbacks.

### **C. Object Detection with Advanced Architectures**
- Train and compare two object detection models:
  1. **Faster-RCNN** (Two-stage object detector).
  2. **YOLOv3** (One-stage object detector).
- Measure detection accuracy using the mAP (mean Average Precision) metric.

## Features
1. **Feature Extraction and Classification (Part A):**
   - Create visual vocabularies using K-Means.
   - Generate histograms for classification using BOVW.
   - Train k-NN and SVM classifiers with varying hyperparameters.

2. **Deep Learning for Classification (Part B):**
   - Custom CNN architecture for multi-class classification.
   - Fine-tuning pre-trained networks for domain-specific tasks.
   - Use of data augmentation techniques for robust training.

3. **Object Detection (Part C):**
   - Train Faster-RCNN and YOLOv3 using the GTSDB dataset.
   - Optimize detection performance using hyperparameter tuning.

## Data
- **Part A:** `caltech-101_5_train` (training) and `caltech-101_5_test` (testing) subsets from the Caltech-101 Dataset.
- **Part B:** `imagedb` (training) and `imagedb_test` (testing) subsets from the Belgian Traffic Sign Dataset.
- **Part C:** GTSDB dataset with train and test subsets.

## Requirements
- Python 3.7+
- Libraries:
  - OpenCV 3.4.2+
  - TensorFlow/Keras
  - NumPy
  - Matplotlib
- Platform: Google Colab (recommended for GPU support).

## Execution
### Part A: Multi-Class Classification (BOVW)
1. Generate visual vocabularies with varying sizes (e.g., 20, 50, 100, 150, 200).
2. Train k-NN and SVM classifiers using BOVW histograms.
3. Evaluate accuracy using the test dataset (`caltech-101_5_test`).

### Part B: Multi-Class Classification (CNNs)
1. **Custom CNN:**
   - Train a CNN from scratch using the `imagedb` dataset.
   - Apply data augmentation for training robustness.
2. **Pre-trained CNN:**
   - Adapt a pre-trained network (e.g., ResNet50) for traffic sign classification.
   - Fine-tune the network for the specific dataset.
3. Evaluate both architectures on the test dataset (`imagedb_test`).

### Part C: Object Detection
1. Train a Faster-RCNN model on the GTSDB dataset.
2. Train a YOLOv3 model using the same dataset.
3. Compare mAP scores for both models.

## Results
Generated outputs and evaluation metrics for all tasks will be stored in the following formats:
- **Part A:** Accuracy results for k-NN and SVM across vocabulary sizes.
- **Part B:** Training and validation accuracy/loss curves for CNN architectures.
- **Part C:** Detection results (bounding boxes) and mAP scores for Faster-RCNN and YOLOv3.

## References
- [Keras Applications](https://keras.io/api/applications/)
- [YOLOv3 Implementation](https://github.com/ultralytics/yolov3)
- [BelgiumTS Dataset](https://btsd.ethz.ch/shareddata/)
- [GTSDB Dataset](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html)

## Notes
1. **Data Augmentation:** Augment training datasets with flipping, rotation, cropping, and zoom.
2. **Input Dimensions:** Ensure all images are resized appropriately (e.g., 64x64 or 128x128 for CNNs).
3. **Batch Size:** Optimize batch size based on GPU memory limits.
4. **Hyperparameter Tuning:** Experiment with k-values for k-NN and hyperparameters for CNNs.
5. **Liscense:** This project is based on an assignment from the "Computer Vision" course at Democritus University of Thrace (DUTH). The original task description is intellectual property of the course instructor.
