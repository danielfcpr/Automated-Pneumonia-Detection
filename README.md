# Automated Pneumonia Detection from X-Rays (Computer Vision)

Early and reliable detection of pneumonia from chest X-rays using a Convolutional Neural Network (Keras/TensorFlow).
This project focuses on recall (sensitivity) to minimize false negatives in a medical context.

# Project Highlights

Task: Binary image classification — normal vs pneumonia

Data size: 5,856 images — 1,583 normal (label=0), 4,273 pneumonia (label=1)

Split: 80% train / 10% val / 10% test (batch size 32)

Model: CNN (Keras / TensorFlow)

Key metric: Recall on pneumonia ≈ 98.5% (priority to reduce false negatives)

Validation approach: Repeated training on different splits to check robustness (principles of cross-validation; not full recorded k-fold)
