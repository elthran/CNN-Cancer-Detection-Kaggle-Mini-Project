# Metastatic Cancer Detection in Digital Pathology Scans

## Overview

This project is aimed at developing an algorithm to identify metastatic cancer in small image patches taken from larger digital pathology scans. The data used for this project is a modified version of the PatchCamelyon (PCam) benchmark dataset.

## Challenge Description

### Problem Statement

The goal is to create an algorithm capable of detecting metastatic cancer within small image patches extracted from whole-slide images (WSI) of lymph node sections. This task involves binary image classification, where each image patch needs to be classified as either containing metastatic tissue or not.

### Relevance

Metastatic cancer detection is a crucial task in medical diagnostics. Automating this process using machine learning can significantly improve the speed and accuracy of diagnosis, aiding pathologists and potentially leading to better patient outcomes.

### Dataset

The dataset provided for this competition is a modified version of the PatchCamelyon (PCam) dataset, which has been curated to remove duplicate images that existed due to probabilistic sampling in the original version. The PCam dataset is known for its balance between task difficulty and tractability, making it suitable for research in various fundamental machine learning topics like active learning, model uncertainty, and explainability.

## Dataset Description

### Size and Structure

The dataset comprises a substantial number of image patches, each measuring 96x96 pixels and containing three color channels (RGB). Each patch is labeled with a binary label indicating the presence (1) or absence (0) of metastatic tissue.

### Data Dimensions

- **Image Patches**: Each image is a 96x96 pixel patch.
- **Color Channels**: Each image has three color channels (RGB).
- **Labels**: Binary labels (0 or 1) indicating the absence or presence of metastatic tissue, respectively.

### Data Distribution

The dataset is balanced, with an equal number of positive (metastatic) and negative (non-metastatic) samples. This balance ensures that models trained on this dataset do not become biased towards either class, which is essential for achieving robust performance in clinical settings.

## Project Structure

### Files and Directories

- `data/`: Contains all the data files.
  - `train/`: Training image patches. **Note: this is not in the git repo due to its size. You can manually download it from kaggle.com**
  - `test/`: Test image patches. **Note: this is not in the git repo due to its size. You can manually download it from kaggle.com**
  - `train_labels.csv`: CSV file containing the labels for the training images.
- `model_trainer.py`: Main script for training and evaluating the model.
- `README.md`: This file.

### Model Training

The project leverages a Convolutional Neural Network (CNN) for the binary classification task. The model is trained using Keras with TensorFlow as the backend. Key features include:

- **Batch Normalization**: Applied after each convolutional layer to stabilize and accelerate the training process.
- **Dropout**: Used to prevent overfitting by randomly dropping units during training.
- **Early Stopping and Learning Rate Reduction**: Implemented as callbacks to optimize the training process.

### Running the Project

To train the model and make predictions, follow these steps:

1. **Prepare the Environment**:
   - Ensure all required libraries are installed using `pip install -r requirements.txt` (eg. `keras`, `tensorflow`, `numpy`, `pandas`, `cv2`).
   - Download and place the dataset in the appropriate directories (`data/train/`, `data/test/`).

2. **Run the Training Script**:
   - Execute `python main.py` to start the training process.
   - The script will train the model, save the best weights, and produce predictions for the test set.

3. **Resume Training**:
   - If training is interrupted, set the `resume` parameter to `True` in the `ModelTrainer` class to resume from the last saved epoch.

**Note**: I had GPU issues on my Mac, so I have a function disable_gpu(). You can set the parameter `disable_gpu` to False to disable this.

## Model and Evaluation

The model architecture includes multiple convolutional layers followed by max pooling, batch normalization, and dropout layers. The final layer is a dense layer with a sigmoid activation function for binary classification. The model is trained using binary cross-entropy loss and Adam optimizer.

### Evaluation Metrics

The primary evaluation metrics are accuracy and loss, monitored on both training and validation sets. The model's performance on the validation set is crucial for detecting overfitting and ensuring generalizability.

## Conclusion

This project demonstrates a practical approach to applying deep learning for metastatic cancer detection in digital pathology scans. By leveraging the PatchCamelyon dataset, we can train effective models that balance complexity and performance, making significant strides towards automated cancer diagnosis in clinical practice.

For more detailed information, refer to the code and comments within the `main.py` script.