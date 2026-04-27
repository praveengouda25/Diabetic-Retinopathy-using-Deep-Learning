<div align="right">
  
[1]: https://github.com/praveengouda25
[2]: https://www.linkedin.com/in/praveen-kumar-bcc2525/

[![github](https://github.com/praveengouda25/Telecom_Customer_Churn_Prediction/blob/4f3921b8f8104e2a1fd9ff8dbf2191765a89e228/icons/git.svg)][1]
[![linkedin](https://github.com/praveengouda25/Telecom_Customer_Churn_Prediction/blob/7cdd63bd3820d6a3cc1d61d0a78f976d942c9ea6/icons/linkedin.svg)][2]

</div>

# Diabetic Retinopathy Detection using Deep Learning
![DR](https://github.com/praveengouda25/Diabetic-Retinopathy-using-Deep-Learning/blob/b7aa21dfe159f1636662d94e4f8df5ef05fa851c/output/Screenshot%20(18).png)) 
## Description
This project detects diabetic retinopathy severity from retinal fundus images using deep learning.  
It includes a trained ResNet50-based classifier and a Gradio web interface for quick local testing and demo use.
Diabetic Retinopathy (DR) is a serious eye disease caused by prolonged diabetes, which damages the blood vessels in the retina and can lead to blindness if not detected early.

- Early detection can prevent severe vision loss
- Manual diagnosis requires expert ophthalmologists
- Automated systems can assist in faster and accessible screening

## Features
- Retinal image classification into 5 diabetic retinopathy severity classes
- ResNet50-based model loading with fallback support
- Interactive Gradio UI for upload-and-predict workflow
- Sample example images for quick testing
- Windows-friendly run scripts and environment setup

## Dataset
![Data-Visualization](https://github.com/praveengouda25/Diabetic-Retinopathy-using-Deep-Learning/blob/a9bec08f81259de3a99391da7fa1156d183d26f1/output/Picture3.png) 
Retinal fundus images dataset
- - Images categorized into 5 classes:
- No_DR
- Mild
- Moderate
- Severe
- Proliferative
Dataset can be sourced from Kaggle or public medical datasets.

### Overview of the Five Categories

The system classifies diabetic retinopathy into five severity levels:

| Category | Label | Description | Clinical Significance |
|----------|-------|-------------|---------------------|
| **No_DR** | 0 | No diabetic retinopathy detected | Normal retina, no treatment needed |
| **Mild** | 1 | Mild non-proliferative diabetic retinopathy | Early stage, monitoring recommended |
| **Moderate** | 2 | Moderate non-proliferative diabetic retinopathy | Moderate damage, regular follow-ups needed |
| **Severe** | 3 | Severe non-proliferative diabetic retinopathy | Significant damage, treatment may be required |
| **Proliferate** | 4 | Proliferative diabetic retinopathy | Advanced stage, urgent treatment required |


## Tech Stack
- Programming: Python
- Libraries: PyTorch, NumPy, Pandas
- Model: ResNet50 (CNN)
- UI Framework: Gradio
  

##  System Architecture

### High-Level System Flow

The system follows a complete machine learning pipeline from data preparation to deployment:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │   Raw Fundus │
    │    Images    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Preprocessing│  ← Crop dark borders, resize, normalize
    │   Pipeline   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   Dataset    │  ← Balance classes, create CSV labels
    │  Balancing   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Train/Val    │  ← 80% training, 20% validation split
    │   Split      │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   Model      │  ← ResNet50 with transfer learning
    │  Training    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Model Saving │  ← Save as dr_model_resnet50_v2.pth
    │   (.pth)     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   Gradio     │  ← Web-based user interface
    │  Interface   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Prediction  │  ← Real-time DR severity classification
    │   Results    │
    └──────────────┘
```

### Detailed Block Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA PREPARATION PHASE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Raw Images → crop_and_resize.py → Balanced Dataset                 │
│      ↓              ↓                        ↓                       │
│  [Various]    [512x512]              [Class-balanced]               │
│  Sizes        Cropped                Folders                         │
│                                                                      │
│  balance_dataset.py → balanced_labels.csv                           │
│      ↓                                                               │
│  [No_DR: 265, Mild: 265, Moderate: 265,                             │
│   Severe: 190, Proliferate: 265]                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          TRAINING PHASE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  train_resnet50.py                                                   │
│      │                                                               │
│      ├─→ Load balanced_labels.csv                                    │
│      ├─→ Split 80/20 (stratified)                                   │
│      ├─→ DataLoader (batch_size=32)                                 │
│      ├─→ ResNet50 Model (pretrained ImageNet)                        │
│      ├─→ Training Loop (5 epochs)                                    │
│      │   ├─→ Forward Pass                                            │
│      │   ├─→ Loss Calculation (CrossEntropyLoss)                    │
│      │   ├─→ Backward Pass                                           │
│      │   └─→ Optimizer Step (Adam, lr=1e-4)                          │
│      └─→ Save Model → dr_model_resnet50_v2.pth                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PHASE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  app.py (Gradio Interface)                                          │
│      │                                                               │
│      ├─→ Load dr_model_resnet50_v2.pth                              │
│      ├─→ User Uploads Fundus Image                                  │
│      ├─→ Preprocessing:                                              │
│      │   ├─→ Resize to 224x224                                       │
│      │   ├─→ Convert to Tensor                                       │
│      │   └─→ Normalize (ImageNet stats)                             │
│      ├─→ Model Forward Pass                                          │
│      ├─→ Softmax → Probabilities                                     │
│      └─→ Display Results (5 class confidences)                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---


### Dataset Structure

The project uses a structured dataset organized into class-specific folders:

```
data/
└── colored_images/
    ├── No_DR/              (Class 0)
    ├── Mild/               (Class 1)
    ├── Moderate/           (Class 2)
    ├── Severe/             (Class 3)
    └── Proliferate_DR/     (Class 4)
```
![Image Processing](https://github.com/praveengouda25/Diabetic-Retinopathy-using-Deep-Learning/blob/1c469112bc2e61b208ba89601320ea4f3f183f34/output/Picture2.png) 
### Class Distribution

**Original Dataset:**
- The original dataset contains fundus images with varying class distributions
- Class imbalance is common in medical datasets, with "No_DR" typically having more samples than severe cases

**Balanced Dataset:**
After running the balancing script (`balance_dataset.py`), the dataset is balanced to:

| Class | Label | Target Count | Final Count |
|-------|-------|--------------|-------------|
| No_DR | 0 | 265 | 265 |
| Mild | 1 | 265 | 265 |
| Moderate | 2 | 265 | 265 |
| Severe | 3 | 190 | 190 |
| Proliferate | 4 | 265 | 265 |

**Total Balanced Images: 1,250**

### Why Balancing is Needed

Class imbalance is a critical problem in medical image classification:

1. **Model Bias**: Without balancing, the model tends to predict the majority class more frequently
2. **Poor Performance on Rare Classes**: Severe and proliferative cases are less common but more critical to detect accurately
3. **Unfair Evaluation**: Accuracy alone becomes misleading when classes are imbalanced
4. **Clinical Impact**: Missing a severe case has worse consequences than misclassifying a mild case

**Solution Implemented:**
- Downsampling: Randomly sample from overrepresented classes
- Augmentation: Generate synthetic images for underrepresented classes using:
  - Random horizontal flips
  - Random rotations (±10 degrees)
  - Color jittering (brightness, contrast, saturation, hue adjustments)

### How Images are Used (Fundus Imaging Basics)

**Fundus Imaging Process:**
1. **Image Capture**: A fundus camera takes a photograph of the retina through a dilated pupil
2. **Image Characteristics**:
   - Circular field of view (typically 30-45 degrees)
   - Dark borders around the circular retina region
   - Central optic disc (bright circular area)
   - Blood vessels radiating from the optic disc
   - Macula (dark central region responsible for sharp vision)

**What the Model Looks For:**
- **Microaneurysms**: Small red dots (early DR sign)
- **Hemorrhages**: Bleeding in the retina
- **Hard Exudates**: Yellow-white deposits
- **Cotton Wool Spots**: White fluffy patches
- **Neovascularization**: Abnormal new blood vessels (proliferative DR)

**Image Preprocessing for Model:**
- Remove dark borders to focus on retinal region
- Resize to standard dimensions (224x224 for ResNet50)
- Normalize pixel values to match ImageNet statistics
- Apply augmentations during training to increase robustness

---

##  Model Training Details

### Model Architecture: ResNet50

**ResNet50** (Residual Network with 50 layers) is a deep convolutional neural network architecture that won the ImageNet competition in 2015. It introduced the concept of "residual connections" or "skip connections" that allow gradients to flow through deeper networks.

**Why ResNet50 was Selected:**

1. **Proven Performance**: Excellent results on ImageNet and medical imaging tasks
2. **Transfer Learning**: Pretrained on ImageNet (1.2 million images), providing strong feature extractors
3. **Depth**: 50 layers provide sufficient capacity to learn complex patterns
4. **Efficiency**: Good balance between accuracy and computational cost
5. **Medical Imaging Success**: Widely used in medical image classification with proven results

---
**Output:**
- Shape: `(1, 5)` - probability distribution over 5 classes
- Values: Between 0.0 and 1.0, sum to 1.0
- Example: `[0.05, 0.10, 0.70, 0.10, 0.05]` → 70% confidence for "Moderate"

### Predicted Class Displayed
![](https://github.com/praveengouda25/Diabetic-Retinopathy-using-Deep-Learning/blob/9762e4f9c2fa0842ea5803697a95cc3a9bb6c6a0/output/Screenshot%20(9).png) 


**Process:**
```python
confidence, predicted_class = torch.max(probabilities, 1)
predicted_class_idx = predicted_class.item()
confidence_score = confidence.item()
```

**Result Extraction:**
- `predicted_class_idx`: Index (0-4) of class with highest probability
- `confidence_score`: Probability value of predicted class
- `confidences`: Dictionary mapping class names to probabilities

**Display Format:**
```python
{
    "No_DR": 0.05,
    "Mild": 0.10,
    "Moderate": 0.70,  # ← Predicted class
    "Severe": 0.10,
    "Proliferate": 0.05
}
```

**User Interface:**
- Gradio displays all 5 class probabilities as a bar chart
- Highlights the predicted class
- Shows confidence percentage
- Provides interpretation guide

![User-Interface](https://github.com/praveengouda25/Diabetic-Retinopathy-using-Deep-Learning/blob/76adc734f58365fbbc1636c1dd3aaafead929996/output/Screenshot%20(15).png) 

### Complete Pipeline Summary

```
User Upload
    ↓
[PIL Image, variable size]
    ↓
Resize to 224×224
    ↓
Convert to Tensor [0, 1]
    ↓
Normalize (ImageNet stats)
    ↓
Add Batch Dimension [1, 3, 224, 224]
    ↓
Model Forward Pass
    ↓
[ResNet50 Feature Extraction]
    ↓
[Fully Connected Layers]
    ↓
Raw Logits [1, 5]
    ↓
Softmax
    ↓
Probabilities [1, 5] (sum = 1.0)
    ↓
Extract Max → Predicted Class
    ↓
Display Results (Gradio UI)
```

**Total Processing Time:**
- Typically 50-200 milliseconds per image (CPU)
- 10-50 milliseconds per image (GPU)
- Depends on image size and hardware

---

##  User Interface (Gradio/Flask UI)

### Purpose of the Interface

The Gradio web interface provides an **accessible, user-friendly platform** for diabetic retinopathy detection without requiring technical expertise. It serves multiple purposes:

1. **Medical Screening**: Enables healthcare workers to quickly screen fundus images
2. **Research Tool**: Allows researchers to test the model on new datasets
3. **Educational**: Demonstrates AI capabilities in medical imaging
4. **Prototype Deployment**: Provides a production-ready interface for clinical trials

---

## Research Publication

This project has been successfully published in a research journal, demonstrating its practical applicability and contribution to the field of medical image analysis.

Title: Diabetic Retinopathy Detection using Deep Learning
Journal: IRJMETS (International Research Journal of Modernization in Engineering Technology and Science)
Year: December, 2025

![Certificate](https://github.com/praveengouda25/Diabetic-Retinopathy-using-Deep-Learning/blob/76adc734f58365fbbc1636c1dd3aaafead929996/output/Screenshot%20(15).png) 

This research presents an automated system for detecting diabetic retinopathy using deep learning techniques. A Convolutional Neural Network based on the ResNet50 architecture is used to analyze retinal fundus images and classify them into different severity levels. The system improves early diagnosis and supports healthcare professionals by providing fast and reliable predictions.

![View-Paper](https://github.com/praveengouda25/Diabetic-Retinopathy-using-Deep-Learning/blob/76adc734f58365fbbc1636c1dd3aaafead929996/output/Screenshot%20(15).png) 

---
## Conclusion ::
This project successfully implements a complete diabetic retinopathy detection system using a Convolutional Neural Network (CNN) model trained on retinal images. Through systematic stages such as data collection, preprocessing, CNN architecture design, model training and evaluation, and deployment through a simple user interface, the system now functions as a reliable automated screening tool. The trained model is capable of accurately classifying diabetic retinopathy severity levels, thereby supporting early diagnosis and reducing manual workload for ophthalmologists. By integrating Python-based machine learning workflows with image-processing techniques, the project demonstrates a fully working solution that enhances the accuracy, accessibility, and efficiency of retinal disease detection.


## Future Enhancement ::
- Integration with a Retinal Image ScannerThe system can be connected to a real-time retinal imaging device to directly capture fundus images and automatically check if the image quality is readable or needs re-capturing before analysis.
  
- Large-Scale Model for Regional AdaptationThe model can be expanded to process large datasets containing images from different regions, age groups, and camera types, making the detection system more accurate for diverse populations.

- Automated Image Quality Assessment ModuleA pre-processing module can be added to automatically detect blurred, under-exposed, or noisy images and prompt the user to retake them, improving overall detection accuracy.

- Advanced Cloud-Based Processing SystemThe system can be deployed on cloud servers to handle large-scale image processing in real time, enabling hospitals and screening camps to analyze thousands of retinal images per day.


## Installation (Windows PowerShell)
```powershell
# 1) Create virtual environment
python -m venv .venv

# 2) Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3) Install dependencies
python -m pip install -r requirements.txt
```

## How to Run the Project
```powershell
# Option A: Direct run
.\.venv\Scripts\python.exe app.py

# Option B: Batch script
.\run.bat
```

After launch, open the URL shown in terminal (for example `http://127.0.0.1:7860`).

## Screenshots
_Add screenshots of the Gradio interface and prediction results here._


