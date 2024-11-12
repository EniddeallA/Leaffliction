# Leaffliction
#### The project is an introduction to Deep Learning/Computer Vision

This project implements a deep learning solution using ResNet18 to classify plant diseases from leaf images, with comprehensive data preprocessing and augmentation pipelines.

## Project Structure
```
./
├── Distribution.py          # Dataset distribution analysis
├── Augmentation.py          # Image augmentation implementation
├── Transformation.py        # PlantCV-based image transformations
├── PlantDiseaseDataset.py   # Custom PyTorch dataset
├── train.py                 # Model training script
├── predict.py               # Disease prediction script
└── Leaves.zip               # Leave image dataset information
```

## Leaves Directory Structure
```
dataset/
├── Plant_Type1/
│   ├── Disease1/
│   │   ├── original_images/
│   │   └── Augmented/
│   │   └── Transformed/
│   └── Disease2/
│       ├── original_images/
│       └── Augmented/
│   │   └── Transformed/
└── Plant_Type2/
    └── ...
```

## Components

### 1. Data Analysis
Our `Distribution.py` script analyzes and visualizes the dataset structure:
```bash
python Distribution.py ./path/to/dataset
```
Features:
- Pie chart visualization of disease distribution
- Bar chart showing image count per disease
- Handles both original and augmented images
- Excludes augmented directories from double-counting

### 2. Data Augmentation
The `Augmentation.py` script implements various augmentation techniques using torchvision:
```bash
python Augmentation.py ./path/to/image.jpg
# or
python Augmentation.py ./path/to/directory
```
Implemented transformations:
- Gaussian Blur (kernel_size=(3, 7), sigma=(1.5, 6))
- Random Rotation (-80° to 80°)
- Color Jitter (brightness, contrast, saturation)
- Random Resized Crop (200x200)
- Random Affine Shear (45°)
- Random Horizontal/Vertical Flip

### 3. Image Transformation
`Transformation.py` uses PlantCV for advanced image processing:
```bash
python Transformation.py ./path/to/image.jpg
# or
python Transformation.py -src source_dir -dst destination_dir
```
Pipeline includes:
- LAB colorspace conversion
- Binary thresholding
- Gaussian blur
- Mask application
- ROI extraction
- Shape analysis
- X/Y axis pseudolandmarks

### 4. Custom Dataset
`PlantDiseaseDataset.py` implements a PyTorch Dataset class with:
- Automatic data balancing
- Support for multiple image formats
- Caching mechanism for faster loading
- Minimum image count validation (500 images)
- Even distribution across classes

### 5. Model Architecture and Training
We use a modified ResNet18 architecture with transfer learning:
```python
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_classes)
```

Training pipeline features:
- 80/20 train-validation split
- Batch size of 32
- Adam optimizer with learning rate 0.0001 and weight decay 1e-5
- CrossEntropyLoss criterion
- Training progress monitoring with loss and accuracy metrics
- Validation accuracy evaluation after each epoch

Image preprocessing pipeline:
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.RandomGrayscale(0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

To train the model:
```bash
python train.py --data_train /path/to/dataset --total_images 1000
```

The trained model is saved as 'plant_disease_model.pth' with both the model state dictionary and class labels.

### 6. Prediction
The `predict.py` script provides disease classification with visualization:
```bash
python predict.py path/to/image1.jpg path/to/image2.jpg ...
```
Features:
- Side-by-side visualization of original and transformed images
- Clear prediction display
- Batch prediction support
- CPU/GPU compatibility
