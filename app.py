import os
import sys
import socket
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from src.model import DRModel
from torchvision import transforms as T, models

# Primary model path - the newly trained ResNet50 model
NEW_MODEL_PATH = "dr_model_resnet50_v2.pth"
MODEL_PATH = "dr_model.pth"  # Legacy model path

# Fallback model paths (for backward compatibility)
ARTIFACTS_DIR = "artifacts"
CKPT_PATH = os.path.join(ARTIFACTS_DIR, "dr-model.ckpt")
PKL_PATH = os.path.join(ARTIFACTS_DIR, "dr-model.pkl")
PT_PATH = os.path.join(ARTIFACTS_DIR, "dr-model.pt")
PTH_PATH = os.path.join(ARTIFACTS_DIR, "dr-model.pth")

# Allow override via environment variable
ENV_MODEL_PATH = os.environ.get("MODEL_PATH", None)


def create_resnet50_model(num_classes=5):
    """Create ResNet50 model with same architecture as training."""
    # Load ResNet50 (weights don't matter for inference, we'll load state_dict)
    model = models.resnet50(weights=None)
    
    # Replace the final fully connected layer to match training architecture
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_features // 2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(num_features // 2, num_classes)
    )
    
    return model


def _normalize_state_dict(state_dict):
    """Normalize state dict keys (remove 'module.' prefix if present)."""
    normalized = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix if present
        new_key = key.replace('module.', '')
        normalized[new_key] = value
    return normalized


def _find_available_port(start_port=7860, end_port=7960):
    """Return the first available localhost TCP port in range."""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError(f"No available port found in range {start_port}-{end_port}")


def _load_model():
    """Load a trained model from dr_model_resnet50_v2.pth (primary) or fallback paths.
    
    This function tries multiple model file formats in order of preference:
    1. dr_model_resnet50_v2.pth (newly trained ResNet50 model)
    2. dr_model.pth (legacy model)
    3. Environment-specified MODEL_PATH
    4. Lightning checkpoint (.ckpt)
    5. PyTorch pickle (.pkl)
    6. PyTorch state dict (.pt/.pth)
    
    Returns:
        Model: Loaded and evaluated model
        
    Raises:
        FileNotFoundError: If no valid model file is found
        RuntimeError: If model loading fails
    """
    # Prefer CPU to avoid CUDA dependency on machines without GPU
    map_location = torch.device("cpu")
    
    # Try the newly trained ResNet50 model first
    if os.path.exists(NEW_MODEL_PATH):
        print(f"[INFO] Attempting to load ResNet50 model: {NEW_MODEL_PATH}")
        try:
            state_dict = torch.load(NEW_MODEL_PATH, map_location=map_location)
            # Normalize state dict keys
            state_dict = _normalize_state_dict(state_dict)
            # Create ResNet50 model with same architecture as training
            model = create_resnet50_model(num_classes=5)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print(f"[OK] Successfully loaded ResNet50 model from: {NEW_MODEL_PATH}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load from {NEW_MODEL_PATH}: {e}")
            import traceback
            traceback.print_exc()
    
    # Try the legacy model (dr_model.pth)
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Attempting to load newly trained model: {MODEL_PATH}")
        try:
            state_dict = torch.load(MODEL_PATH, map_location=map_location)
            # Create model instance with same architecture as training
            model = DRModel(num_classes=5, model_name="densenet121")
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print(f"[OK] Successfully loaded model from: {MODEL_PATH}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load from {MODEL_PATH}: {e}")
            import traceback
            traceback.print_exc()
    
    # Try environment-specified path
    if ENV_MODEL_PATH and os.path.exists(ENV_MODEL_PATH):
        print(f"[INFO] Attempting to load model from environment-specified path: {ENV_MODEL_PATH}")
        try:
            if ENV_MODEL_PATH.lower().endswith(".ckpt"):
                model = DRModel.load_from_checkpoint(ENV_MODEL_PATH, map_location=map_location)
                model.eval()
                print(f"[OK] Successfully loaded Lightning checkpoint: {ENV_MODEL_PATH}")
                return model
            else:
                obj = torch.load(ENV_MODEL_PATH, map_location=map_location)
                if isinstance(obj, DRModel) or hasattr(obj, "forward"):
                    model = obj
                else:
                    model = DRModel(num_classes=5, model_name="densenet121")
                    model.load_state_dict(obj, strict=False)
                model.eval()
                print(f"[OK] Successfully loaded PyTorch model: {ENV_MODEL_PATH}")
                return model
        except Exception as e:
            print(f"[ERROR] Failed to load from {ENV_MODEL_PATH}: {e}")
    
    # Try Lightning checkpoint
    if os.path.exists(CKPT_PATH):
        print(f"Attempting to load Lightning checkpoint: {CKPT_PATH}")
        try:
            # Try loading as checkpoint first
            try:
                model = DRModel.load_from_checkpoint(CKPT_PATH, map_location=map_location)
            except Exception:
                # If checkpoint loading fails, try loading as regular PyTorch file
                checkpoint = torch.load(CKPT_PATH, map_location=map_location)
                if isinstance(checkpoint, DRModel):
                    model = checkpoint
                elif "state_dict" in checkpoint:
                    model = DRModel(num_classes=5, model_name="densenet121")
                    model.load_state_dict(checkpoint["state_dict"], strict=False)
                else:
                    model = DRModel(num_classes=5, model_name="densenet121")
                    model.load_state_dict(checkpoint, strict=False)
            
            model.eval()
            print(f"[OK] Successfully loaded Lightning checkpoint: {CKPT_PATH}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load Lightning checkpoint: {e}")
    
    # Try PyTorch pickle file
    if os.path.exists(PKL_PATH):
        print(f"Attempting to load PyTorch pickle: {PKL_PATH}")
        try:
            obj = torch.load(PKL_PATH, map_location=map_location)
            if isinstance(obj, DRModel) or hasattr(obj, "forward"):
                model = obj
            else:
                model = DRModel(num_classes=5, model_name="densenet121")
                model.load_state_dict(obj, strict=False)
            model.eval()
            print(f"[OK] Successfully loaded PyTorch pickle: {PKL_PATH}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load PyTorch pickle: {e}")
    
    # Try PyTorch state dict files
    for path in [PT_PATH, PTH_PATH]:
        if os.path.exists(path):
            print(f"Attempting to load PyTorch state dict: {path}")
            try:
                obj = torch.load(path, map_location=map_location)
                if isinstance(obj, DRModel) or hasattr(obj, "forward"):
                    model = obj
                else:
                    model = DRModel(num_classes=5, model_name="densenet121")
                    model.load_state_dict(obj, strict=False)
                model.eval()
                print(f"[OK] Successfully loaded PyTorch state dict: {path}")
                return model
            except Exception as e:
                print(f"[ERROR] Failed to load PyTorch state dict from {path}: {e}")
    
    # If we get here, no model was found
    print("\n" + "="*60)
    print("[ERROR] NO MODEL FILE FOUND!")
    print("="*60)
    print("Please ensure one of the following model files exists:")
    print(f"  - {NEW_MODEL_PATH} (Primary: ResNet50 model)")
    print(f"  - {MODEL_PATH} (Legacy: DenseNet121 model)")
    print(f"  - {CKPT_PATH} (Lightning checkpoint)")
    print(f"  - {PKL_PATH} (PyTorch pickle)")
    print(f"  - {PT_PATH} (PyTorch state dict)")
    print(f"  - {PTH_PATH} (PyTorch state dict)")
    print("\nOr set MODEL_PATH environment variable to your model file:")
    print("  Windows: $env:MODEL_PATH=\"path\\to\\your\\model.pth\"")
    print("  Linux/Mac: export MODEL_PATH=\"path/to/your/model.pth\"")
    print("="*60)
    
    raise FileNotFoundError(
        f"No valid model file found. Please ensure '{NEW_MODEL_PATH}' exists in the root directory "
        f"or place a model file in the '{ARTIFACTS_DIR}' directory."
    )


# Load the model
print("[INFO] Searching for model files...")
model = _load_model()
model.eval()  # Ensure model is in evaluation mode for inference
print("[OK] Model ready for inference")

# Class labels matching the training dataset
labels = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate",
}

# Image preprocessing pipeline matching the validation transform from training
# This ensures consistent preprocessing between training and inference
transform = T.Compose([
    T.Resize((224, 224)),  # Resize to 224x224 (matching training)
    T.ToTensor(),  # Convert PIL Image to tensor and scale to [0, 1]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])


# Define the prediction function
def predict(input_img):
    """Predict diabetic retinopathy severity from an input image.
    
    Args:
        input_img: PIL Image from Gradio
        
    Returns:
        dict: Dictionary with class labels and confidence scores
    """
    try:
        # Convert image to RGB if it's not already (handles RGBA, L, etc.)
        if input_img.mode != "RGB":
            input_img = input_img.convert("RGB")
        
        # Apply preprocessing transforms (matches training validation transform)
        input_tensor = transform(input_img).unsqueeze(0)  # Add batch dimension
        
        # Make prediction with model in eval mode
        with torch.no_grad():
            logits = model(input_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_class_idx = predicted_class.item()
            confidence_score = confidence.item()
            
            # Create dictionary with all class probabilities
            confidences = {labels[i]: float(probabilities[0][i]) for i in labels}
            
            # Add prediction info for debugging
            print(f"Predicted class: {labels[predicted_class_idx]} (index: {predicted_class_idx})")
            print(f"Confidence: {confidence_score:.4f}")
            print(f"All probabilities: {confidences}")
        
        return confidences
        
    except Exception as e:
        print(f"[ERROR] Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return {"Error": f"Prediction failed: {str(e)}"}


# Set up the Gradio app interface with professional UI
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Diabetic Retinopathy Detection",
    css="""
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .description-text {
        text-align: center;
        padding: 0 2rem 1.5rem 2rem;
        color: #666;
        line-height: 1.6;
    }
    .instruction-text {
        text-align: center;
        padding: 1rem 0;
        font-weight: 500;
        color: #333;
    }
    .card-container {
        padding: 2rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    """
) as dr_app:
    # Header Section
    gr.Markdown(
        """
        <div class="main-header">
            <h1>🔬 Diabetic Retinopathy Detection App</h1>
        </div>
        """,
        elem_classes=["main-header"]
    )
    
    gr.Markdown(
        """
        <div class="description-text">
            <p>
                Welcome to our Diabetic Retinopathy Detection App! This app utilizes deep learning models 
                to detect diabetic retinopathy in retinal images. Diabetic retinopathy is a common complication 
                of diabetes and early detection is crucial for effective treatment.
            </p>
        </div>
        """,
        elem_classes=["description-text"]
    )
    
    gr.Markdown(
        """
        <div class="instruction-text">
            📤 Upload a retinal fundus image to get prediction.
        </div>
        """,
        elem_classes=["instruction-text"]
    )
    
    # Main Content Section
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                input_image = gr.Image(
                    type="pil",
                    label="Retinal Fundus Image",
                    height=400
                )
                submit_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg"
                )
            
            # Example images section
            gr.Markdown("### 📋 Example Images")
            gr.Examples(
                examples=[
                    os.path.join("data", "colored_images", "Examples", "MODERATE_0a1076183736.png"),
                    os.path.join("data", "colored_images", "Examples", "NO-DR_ 00cc2b75cddd.png"),
                    os.path.join("data", "colored_images", "Examples", "PROLIFERATE_f58d37d48e42.png"),
                    os.path.join("data", "colored_images", "Examples", "SEVERE_913490237ad4.png"),
                ],
                inputs=input_image,
                label="Click on an example image to load it"
            )
        
        with gr.Column(scale=1):
            with gr.Group():
                output_label = gr.Label(
                    label="Prediction Results",
                    num_top_classes=5,
                    container=True,
                    show_label=True
                )
            
            gr.Markdown(
                """
                ### 📊 Results Interpretation
                
                The model predicts the severity of diabetic retinopathy:
                - **No_DR**: No diabetic retinopathy detected
                - **Mild**: Mild non-proliferative diabetic retinopathy
                - **Moderate**: Moderate non-proliferative diabetic retinopathy
                - **Severe**: Severe non-proliferative diabetic retinopathy
                - **Proliferate**: Proliferative diabetic retinopathy
                
                *Higher confidence scores indicate more certain predictions.*
                """
            )
    
    # Connect the prediction function
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=output_label
    )
    
    # Also trigger on image upload/change for convenience
    input_image.upload(
        fn=predict,
        inputs=input_image,
        outputs=output_label
    )

# Run the Gradio app
if __name__ == "__main__":
    server_name = os.environ.get("HOST", "127.0.0.1")
    # Use configured port or auto-find a free port in common Gradio range.
    port_env = os.environ.get("PORT", None)
    server_port = int(port_env) if port_env else _find_available_port(7860, 7960)
    
    print(f"[INFO] Starting Gradio server on {server_name}...")
    print("   If port is busy, Gradio will automatically find an available port.")
    dr_app.launch(server_name=server_name, server_port=server_port, share=False, show_error=True)
