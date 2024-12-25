
# Lung Nodule Detection using Segmentation-Based V-Net Architecture

## Overview
This project is a solo endeavor aimed at automating the detection of lung nodules from volumetric CT scans using a segmentation-based approach with a 3D V-Net architecture. The primary objective is to precisely segment lung regions and identify potential nodules to aid in early detection and diagnosis.

The project leverages advanced preprocessing techniques, efficient patch-based segmentation, and deep learning to address the challenges posed by medical image analysis. The solution was developed and tested on the LUNA16 dataset, showcasing promising results.

---

## Features
1. **Preprocessing Pipeline**:
   - Noise reduction and contrast enhancement for clearer visualization.
   - Extraction of Regions of Interest (ROIs) to isolate relevant lung regions.
   - Conversion of 3D CT scans into 2D slices and patch generation.

2. **Model**:
   - **V-Net Architecture**:
     - Encoder-decoder structure with 3D convolutions.
     - Dice coefficient-based loss function for effective handling of class imbalance.
     - Xavier initialization for gradient stability.
   - Trained on 311 CT scans, generating 10,930 patches.

3. **Evaluation Metrics**:
   - Dice Coefficient
   - Intersection-over-Union (IoU)
   - Sensitivity

4. **Results**:
   - Achieved high training accuracy of 91.6%.
   - Demonstrated low false-positive rates during testing.
   - Results show promise for clinical applications, with opportunities for further refinement.

---

## Installation

### Prerequisites
Ensure the following software and hardware requirements are met:
- Python 3.8 or higher
- CUDA-enabled GPU (Recommended: NVIDIA RTX 3070 or equivalent)
- 32GB RAM or higher
- Operating System: Linux/Windows

### Dependencies
Install the necessary Python libraries using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Additional Tools
- [ITK-SNAP](http://www.itksnap.org/) for visualizing segmentation results.
- GPU drivers and CUDA toolkit for accelerating training.

---

## Dataset
The project uses the **LUNA16 dataset** for training and testing. Ensure you download the dataset from the official [LUNA16 website](https://luna16.grand-challenge.org/) and structure it as follows:

```
data/
├── images/
│   ├── scan_001.mhd
│   ├── scan_002.mhd
│   └── ...
├── masks/
│   ├── mask_001.mhd
│   ├── mask_002.mhd
│   └── ...
└── annotations.csv
```

---

## Preprocessing
1. **Run `mask_data_extraction.py`**:
   - Converts annotations to binary masks using world-to-voxel coordinate conversion.
   - Masks are saved in the `masks/` directory.

2. **Run `slice_splitting.py`**:
   - Splits 3D scans and masks into 2D slices.
   - Normalizes intensity values and extracts ROIs.
   - Output stored as BMP files in `processed_slices/`.

3. **Run `segmentation.py`**:
   - Generates patches using a sliding window approach.
   - Removes irrelevant background elements.
   - Output stored in `patches/`.

---

## Training the Model
1. Configure the paths and parameters in `model_train.py`.
2. Start training:
   ```bash
   python model_train.py
   ```
3. **Training Configuration**:
   - Input patch size: 96x96x16.
   - Epochs: 7000.
   - Batch size: 6.
   - Optimizer: Adam.
   - Loss Function: Dice coefficient-based.

4. Training logs and model checkpoints will be saved in the `checkpoints/` directory.

---

## Testing the Model
1. Use `testing_script.py` to evaluate the model:
   ```bash
   python testing_script.py
   ```
2. **Functionality**:
   - Splits test images into patches.
   - Predicts segmentation for each patch using the trained V-Net.
   - Displays segmented regions in a grid.

3. Results include:
   - Dice Coefficient
   - IoU
   - Sensitivity

---

## Results
1. **Training**:
   - Achieved 91.6% accuracy on training data.
   - Low false positives and consistent convergence over epochs.

2. **Testing**:
   - The model performed well on unseen data, accurately identifying nodules in test images.
   - Metrics:
     - Dice Coefficient: ~0.89
     - IoU: ~0.81
     - Sensitivity: ~0.92

3. **Visualization**:
   - Predicted segmentations are visualized with tumor locations highlighted.
   - Example results are saved in the `results/` directory.

---

## File Structure
```
Lung_Segmentation_Vnet/
├── Architecture/
│   ├── __init__.py
│   ├── layer.py
│   └── vnet_architecture.py
├── data_preprocessing/
│   ├── __init__.py
│   ├── mask_data_extraction.py
│   ├── segmentation.py
│   ├── slice_splitting.py
│   └── utils.py
├── annotations.csv
├── train_X.csv
├── train_X_mask.csv
├── model_train.py
├── model_test.py
├── testing_script.py
├── superimpose.py
├── requirements.txt
└── README.md
```

---

## Future Work
- Expand the training dataset to the full LUNA16 set for better generalization.
- Incorporate advanced attention mechanisms to enhance feature extraction.
- Explore post-processing techniques to reduce false positives further.

---

## Conclusion
This project demonstrates the potential of segmentation-based approaches for lung nodule detection using a V-Net architecture. By combining robust preprocessing, state-of-the-art architecture, and detailed evaluation, this work lays a foundation for scalable and accurate medical imaging solutions.

For questions or feedback, feel free to reach out via GitHub Issues.
