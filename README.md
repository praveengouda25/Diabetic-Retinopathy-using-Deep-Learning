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
![](https://github.com/praveengouda25/Diabetic-Retinopathy-using-Deep-Learning/blob/e1a6304e03150d250477461130a47d741542024d/output/Screenshot%20(9).png) 

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


