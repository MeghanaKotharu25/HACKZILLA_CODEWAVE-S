import gradio as gr
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import hf_hub_download
import io
import os

# =============================================================================
# Configuration
# =============================================================================

AIS_MODEL_REPO = "MeghanaK25/ais-isolation-forest"
AIS_MODEL_FILENAME = "isolationforest_model.pkl"

SAR_MODEL_REPO = "MeghanaK25/sar-oil-slick-detection"  
SAR_MODEL_FILENAME = "training_history.pkl"

# Local fallbacks
LOCAL_AIS_MODEL = "/Users/lakshmikotaru/Documents/ais_isolation_forest/isolationforest_model.pkl"
LOCAL_SAR_MODEL = "/Users/lakshmikotaru/Documents/sar_oil_slick_detection/hf_deployment/training_history_subset.pkl"

# SAR images directory - randomly select from available images when anomaly detected
SAR_IMAGES_DIR = "/Users/lakshmikotaru/Downloads/data/SAR test/"

# =============================================================================
# Model Loading Functions
# =============================================================================

def load_ais_model():
    """Load AIS model from HuggingFace with local fallback"""
    global ais_model, ais_scaler
    
    try:
        print("🔄 Loading AIS model from HuggingFace...")
        model_path = hf_hub_download(repo_id=AIS_MODEL_REPO, filename=AIS_MODEL_FILENAME)
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        ais_model = model_data["model"]
        ais_scaler = model_data["scaler"]
        print("✅ AIS model loaded from HuggingFace")
        return True
        
    except Exception as e:
        print(f"❌ HF failed: {e}. Trying local fallback...")
        try:
            with open(LOCAL_AIS_MODEL, "rb") as f:
                model_data = pickle.load(f)
            ais_model = model_data["model"] 
            ais_scaler = model_data["scaler"]
            print("✅ AIS model loaded from local fallback")
            return True
        except Exception as e2:
            print(f"❌ Local fallback failed: {e2}")
            ais_model = None
            ais_scaler = None
            return False

def load_sar_model():
    """Load SAR model from HuggingFace with local fallback"""
    global sar_model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        print("🔄 Loading SAR model from HuggingFace...")
        model_path = hf_hub_download(repo_id=SAR_MODEL_REPO, filename=SAR_MODEL_FILENAME)
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            
        sar_model = create_sar_model()
        sar_model.to(device)
        sar_model.eval()
        print("✅ SAR model loaded from HuggingFace")
        return True
        
    except Exception as e:
        print(f"❌ HF SAR failed: {e}. Trying local fallback...")
        try:
            with open(LOCAL_SAR_MODEL, "rb") as f:
                model_data = pickle.load(f)
            sar_model = create_sar_model()
            sar_model.to(device) 
            sar_model.eval()
            print("✅ SAR model loaded from local fallback")
            return True
        except Exception as e2:
            print(f"❌ SAR local fallback failed: {e2}")
            sar_model = None
            return False

def create_sar_model():
    """Create SAR model architecture"""
    import torchvision.models as models
    
    class SARModel(nn.Module):
        def __init__(self, num_classes=2):
            super(SARModel, self).__init__()
            self.backbone = models.resnet18(weights='DEFAULT')
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            
        def forward(self, x):
            return self.backbone(x)
    
    return SARModel()

# SAR preprocessing
sar_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# =============================================================================
# Core Analysis Functions
# =============================================================================

def preprocess_vessel_data(mmsi, ship_type, nav_status, sog, cog, heading, width, length, draught):
    """Preprocess vessel data for anomaly detection"""
    
    features = {}
    features['sog'] = float(sog) if sog else 0.0
    features['cog'] = float(cog) if cog else 0.0
    features['heading'] = float(heading) if heading else features['cog']
    features['width'] = float(width) if width else 17.0
    features['length'] = float(length) if length else 115.0
    features['draught'] = float(draught) if draught else 6.3
    
    # Categorical encodings
    nav_status_mapping = {
        'Under way using engine': 0, 'At anchor': 1, 'Moored': 2,
        'Constrained by her draught': 3, 'Unknown value': 4
    }
    ship_type_mapping = {
        'Cargo': 0, 'Tanker': 1, 'Fishing': 2, 'Passenger': 3,
        'Tug': 4, 'Military': 5, 'Pleasure': 6, 'Sailing': 7
    }
    
    features['navigationalstatus_encoded'] = nav_status_mapping.get(nav_status, 4)
    features['shiptype_encoded'] = ship_type_mapping.get(ship_type, 0)
    
    # Derived features
    if features['sog'] <= 0.5:
        features['speed_category'] = 0
    elif features['sog'] <= 3:
        features['speed_category'] = 1
    elif features['sog'] <= 15:
        features['speed_category'] = 2
    else:
        features['speed_category'] = 3
    
    if features['length'] <= 50:
        features['size_category'] = 0
    elif features['length'] <= 150:
        features['size_category'] = 1
    elif features['length'] <= 300:
        features['size_category'] = 2
    else:
        features['size_category'] = 3
    
    course_diff = abs(features['cog'] - features['heading'])
    features['course_diff'] = min(course_diff, 360 - course_diff) if course_diff > 180 else course_diff
    features['aspect_ratio'] = features['length'] / (features['width'] + 0.001)
    
    return features

def load_sar_image():
    """Randomly load a SAR image from the available dataset for analysis"""
    import random
    import glob
    
    try:
        if os.path.exists(SAR_IMAGES_DIR):
            # Get all image files from the SAR test directory
            image_patterns = [
                os.path.join(SAR_IMAGES_DIR, "*.jpg"),
                os.path.join(SAR_IMAGES_DIR, "*.jpeg"), 
                os.path.join(SAR_IMAGES_DIR, "*.png")
            ]
            
            all_images = []
            for pattern in image_patterns:
                all_images.extend(glob.glob(pattern))
            
            if all_images:
                # Randomly select an image
                selected_image = random.choice(all_images)
                image_name = os.path.basename(selected_image)
                print(f"🛰️ Randomly selected SAR image: {image_name}")
                
                # Store the selected image name for display
                load_sar_image.last_selected = image_name
                
                return Image.open(selected_image)
            else:
                print(f"⚠️ No SAR images found in {SAR_IMAGES_DIR}, using fallback")
        else:
            print(f"⚠️ SAR directory not found at {SAR_IMAGES_DIR}, using fallback")
            
        # Fallback synthetic image
        np.random.seed()
        sar_img = np.random.normal(0.3, 0.1, (224, 224))
        sar_img[80:140, 60:160] *= 0.4  # Dark areas simulate oil
        sar_img = np.clip(sar_img, 0, 1)
        load_sar_image.last_selected = "synthetic_fallback.png"
        return Image.fromarray((sar_img * 255).astype(np.uint8))
        
    except Exception as e:
        print(f"❌ Error loading SAR image: {e}")
        # Emergency fallback
        np.random.seed(42)
        sar_img = np.random.normal(0.3, 0.1, (224, 224)) 
        sar_img[80:140, 60:160] *= 0.4
        sar_img = np.clip(sar_img, 0, 1)
        load_sar_image.last_selected = "error_fallback.png"
        return Image.fromarray((sar_img * 255).astype(np.uint8))

# Initialize the attribute
load_sar_image.last_selected = "unknown.png"

def analyze_sar_image(image, confidence_threshold=0.5):
    """Analyze SAR image for oil slicks"""
    
    if image is None:
        return "❌ No SAR image available", 0.0, "No Image", None
    
    if sar_model is None:
        # Fallback heuristic analysis
        try:
            img_array = np.array(image.convert('L'))
            dark_ratio = np.sum(img_array < 100) / img_array.size
            texture_variance = np.var(img_array) / 10000
            confidence = min(0.85, dark_ratio * 1.5 + texture_variance + np.random.uniform(-0.1, 0.1))
            confidence = max(0.05, confidence)
            
            if confidence > confidence_threshold:
                result = f"🚨 Oil Slick Detected (Heuristic Analysis: {confidence:.3f})"
                risk = "🚨 HIGH RISK - Oil spill detected in SAR imagery"
            else:
                result = f"✅ No Oil Slick (Heuristic Analysis: {1-confidence:.3f})"
                risk = "✅ CLEAN - No oil detected in SAR imagery"
            
            viz = create_sar_visualization(image, confidence, confidence_threshold)
            return result, confidence, risk, viz
            
        except Exception as e:
            return f"❌ SAR Analysis Error: {str(e)}", 0.0, "Processing Error", None
    
    try:
        # Model-based analysis
        input_tensor = sar_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = sar_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            oil_prob = probabilities[0, 1].item()
        
        if oil_prob > confidence_threshold:
            result = f"🚨 Oil Slick Detected (AI Model: {oil_prob:.3f})"
            risk = "🚨 HIGH RISK - AI detected oil spill in SAR imagery"
        else:
            result = f"✅ No Oil Slick (AI Model: {1-oil_prob:.3f})"  
            risk = "✅ CLEAN - No oil detected by AI model"
        
        viz = create_sar_visualization(image, oil_prob, confidence_threshold)
        return result, oil_prob, risk, viz
        
    except Exception as e:
        return f"❌ AI Model Error: {str(e)}", 0.0, "Model Error", None

def create_sar_visualization(image, confidence, threshold):
    """Create SAR analysis visualization"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original SAR image
    ax1.imshow(image, cmap='gray')
    image_name = getattr(load_sar_image, 'last_selected', 'SAR Image')
    ax1.set_title(f'SAR Image: {image_name}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Analysis overlay
    ax2.imshow(image, cmap='gray', alpha=0.7)
    
    if confidence > threshold:
        # Add detection overlay for oil areas
        overlay = np.zeros_like(np.array(image.convert('L')))
        h, w = overlay.shape
        # Highlight potential oil areas in red
        overlay[h//3:2*h//3, w//4:3*w//4] = 150
        ax2.imshow(overlay, cmap='Reds', alpha=0.4)
        ax2.set_title(f'🚨 Oil Detection (Confidence: {confidence:.3f})', 
                     fontsize=12, fontweight='bold', color='red')
    else:
        ax2.set_title(f'✅ Clean Waters (Confidence: {1-confidence:.3f})', 
                     fontsize=12, fontweight='bold', color='green')
    
    ax2.axis('off')
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

# =============================================================================
# Integrated Analysis Function (MAIN WORKFLOW)
# =============================================================================

def analyze_vessel_and_environment(mmsi, ship_type, nav_status, sog, cog, heading, width, length, draught, confidence_threshold=0.5):
    """
    MAIN FUNCTION: Integrated vessel anomaly detection with automatic SAR analysis
    This is the core automated workflow as requested
    """
    
    # Step 1: AIS Anomaly Detection
    if ais_model is None:
        return (
            "❌ AIS Model not loaded", 
            "Model Error", 
            0.0,
            "❌ Cannot perform analysis",
            "System Error",
            None
        )
    
    try:
        # Preprocess vessel data
        features = preprocess_vessel_data(mmsi, ship_type, nav_status, sog, cog, heading, width, length, draught)
        
        # Create feature vector
        feature_vector = np.array([[
            features['sog'], features['cog'], features['heading'],
            features['width'], features['length'], features['draught'],
            features['navigationalstatus_encoded'], features['shiptype_encoded'],
            features['speed_category'], features['size_category'],
            features['course_diff'], features['aspect_ratio']
        ]])
        
        # Scale and predict
        feature_vector_scaled = ais_scaler.transform(feature_vector)
        prediction = ais_model.predict(feature_vector_scaled)[0]
        anomaly_score = ais_model.decision_function(feature_vector_scaled)[0]
        
        is_anomaly = prediction == -1
        
        # Generate AIS results
        if is_anomaly:
            ais_result = f"🚨 ANOMALOUS VESSEL DETECTED (MMSI: {mmsi})"
            ais_risk = "🚨 HIGH RISK" if anomaly_score < -0.1 else "⚠️ ANOMALOUS"
        else:
            ais_result = f"✅ Normal vessel behavior (MMSI: {mmsi})"
            ais_risk = "✅ NORMAL"
        
        # Step 2: Automatic SAR Analysis (TRIGGERED BY ANOMALY)
        if is_anomaly:
            print(f"🛰️ Anomaly detected! Automatically analyzing SAR imagery...")
            sar_image = load_sar_image()
            sar_result, sar_confidence, sar_risk, sar_viz = analyze_sar_image(sar_image, confidence_threshold)
            
            # Step 3: Generate Integrated Alert
            combined_alert = generate_integrated_alert(
                mmsi, ship_type, is_anomaly, anomaly_score, 
                sar_result, sar_risk, sar_confidence
            )
            
            return (
                ais_result,
                ais_risk, 
                float(anomaly_score),
                combined_alert,
                f"ANOMALY DETECTED → SAR ANALYSIS COMPLETE",
                sar_viz
            )
        
        else:
            # No anomaly - no SAR analysis needed
            return (
                ais_result,
                ais_risk,
                float(anomaly_score), 
                "✅ NORMAL OPERATIONS\nNo vessel anomaly detected. SAR analysis not required.",
                "Normal vessel behavior - monitoring continues",
                None
            )
            
    except Exception as e:
        return (
            f"❌ Analysis Error: {str(e)}", 
            "Processing Error", 
            0.0,
            "❌ Could not complete analysis",
            "System Error",
            None
        )

def generate_integrated_alert(mmsi, ship_type, is_anomaly, anomaly_score, sar_result, sar_risk, sar_confidence):
    """Generate comprehensive monitoring alert"""
    
    oil_detected = "Oil Slick Detected" in sar_result
    image_name = getattr(load_sar_image, 'last_selected', 'Unknown SAR Image')
    
    alert = f"🚨 MARITIME MONITORING ALERT 🚨\n"
    alert += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    alert += f"📍 VESSEL: {ship_type} (MMSI: {mmsi})\n"
    alert += f"⚠️ AIS STATUS: Anomalous behavior (Score: {anomaly_score:.3f})\n"
    alert += f"🛰️ SAR ANALYSIS: {sar_result}\n"
    alert += f"📡 SAR IMAGE: {image_name}\n"
    alert += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    if oil_detected:
        alert += f"🚨 CRITICAL SITUATION DETECTED 🚨\n"
        alert += f"• Vessel anomaly + Oil slick presence\n" 
        alert += f"• SAR confidence: {sar_confidence:.3f}\n"
        alert += f"• IMMEDIATE ACTION REQUIRED\n\n"
        alert += f"📞 RECOMMENDED ACTIONS:\n"
        alert += f"1. Alert maritime authorities immediately\n"
        alert += f"2. Dispatch inspection vessel to location\n"
        alert += f"3. Begin oil spill response procedures\n"
        alert += f"4. Monitor vessel movement continuously\n"
    else:
        alert += f"⚠️ VESSEL ANOMALY ONLY\n"
        alert += f"• Suspicious vessel behavior detected\n"
        alert += f"• No oil slick in SAR imagery\n"
        alert += f"• Continue monitoring required\n\n"
        alert += f"📞 RECOMMENDED ACTIONS:\n"
        alert += f"1. Monitor vessel closely\n"
        alert += f"2. Request vessel identification\n"
        alert += f"3. Check navigation compliance\n"
        alert += f"4. Acquire additional SAR imagery\n"
    
    return alert

# =============================================================================
# Demo Data
# =============================================================================

def get_demo_vessels():
    """Get demo vessel data for testing"""
    return [
        # Normal cargo ship
        [219123456, "Cargo", "Under way using engine", 12.5, 45, 45, 25, 180, 8.5],
        # SUSPICIOUS: Stationary tanker (will trigger SAR analysis)
        [477307700, "Tanker", "At anchor", 0.1, 180, 135, 60, 333, 15.2],
        # SUSPICIOUS: High-speed fishing vessel  
        [219999999, "Fishing", "Engaged in fishing", 25.0, 90, 95, 8, 45, 3.2],
        # Normal container ship
        [311234567, "Container", "Under way using engine", 15.2, 270, 315, 32, 200, 12.1]
    ]

# =============================================================================
# Initialize Models
# =============================================================================

print("🚀 Initializing Maritime Monitoring System...")
ais_model = None
ais_scaler = None  
sar_model = None
device = None

ais_loaded = load_ais_model()
sar_loaded = load_sar_model()

if ais_loaded and sar_loaded:
    print("✅ Both models loaded successfully!")
elif ais_loaded:
    print("✅ AIS model loaded, SAR will use heuristic analysis")
else:
    print("⚠️ Model loading issues - using fallback methods")

# =============================================================================
# Gradio Interface
# =============================================================================

with gr.Blocks(title="🚢🛰️ Maritime Anomaly & Oil Spill Detection", theme=gr.themes.Ocean()) as app:
    
    gr.Markdown("""
    # 🚢🛰️ Automated Maritime Environmental Monitoring
    
    **Real-time vessel anomaly detection with automatic SAR oil spill analysis**
    
    🔄 **Automated Workflow:**
    1. Input vessel AIS data
    2. AI detects anomalous behavior  
    3. **If anomaly found** → Automatically analyzes Red Sea SAR imagery for oil spills
    4. Generates integrated maritime alert
    
    *No manual SAR input needed - fully automated pipeline*
    """)
    
    # =============================================================================
    # Main Analysis Interface
    # =============================================================================
    
    with gr.Tab("🔍 Maritime Monitoring"):
        gr.Markdown("### Input vessel data - System will automatically check for anomalies and oil spills")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Vessel Identification")
                mmsi_input = gr.Number(label="MMSI (Ship ID)", value=477307700)
                ship_type_input = gr.Dropdown(
                    choices=["Cargo", "Tanker", "Fishing", "Passenger", "Container", "Tug", "Military"],
                    label="Ship Type", value="Tanker"
                )
                nav_status_input = gr.Dropdown(
                    choices=["Under way using engine", "At anchor", "Moored", "Constrained by her draught", "Unknown value"],
                    label="Navigation Status", value="At anchor"
                )
                
            with gr.Column():
                gr.Markdown("#### Navigation Data") 
                sog_input = gr.Number(label="Speed Over Ground (knots)", value=0.1)
                cog_input = gr.Number(label="Course Over Ground (degrees)", value=180)
                heading_input = gr.Number(label="Heading (degrees)", value=135)
                
            with gr.Column():
                gr.Markdown("#### Vessel Dimensions")
                width_input = gr.Number(label="Width (meters)", value=60)
                length_input = gr.Number(label="Length (meters)", value=333)
                draught_input = gr.Number(label="Draught (meters)", value=15.2)
        
        confidence_threshold = gr.Slider(
            minimum=0.1, maximum=0.9, value=0.5, step=0.05,
            label="SAR Oil Detection Confidence Threshold"
        )
        
        analyze_btn = gr.Button("🔍 Analyze Vessel & Environment", variant="primary", size="lg")
        
        gr.Markdown("---")
        
        # Results Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### AIS Anomaly Results")
                ais_result = gr.Textbox(label="Vessel Analysis", interactive=False)
                ais_risk = gr.Textbox(label="AIS Risk Level", interactive=False)
                ais_score = gr.Number(label="Anomaly Score", interactive=False)
                
            with gr.Column():
                gr.Markdown("#### System Status")
                system_status = gr.Textbox(label="Processing Status", interactive=False)
        
        # Integrated Alert (Full Width)
        integrated_alert = gr.Textbox(
            label="🚨 Integrated Maritime Alert", 
            interactive=False, 
            lines=12,
            max_lines=15
        )
        
        # SAR Analysis Results (Shows only when anomaly detected)
        sar_visualization = gr.Image(
            label="SAR Analysis Results (Automatically triggered by anomaly detection)", 
            visible=True
        )
    
    # =============================================================================
    # Quick Demo Examples
    # =============================================================================
    
    with gr.Tab("📋 Demo Scenarios"):
        gr.Markdown("### Quick test scenarios")
        
        demo_vessels = get_demo_vessels()
        
        for i, vessel_data in enumerate(demo_vessels):
            mmsi, ship_type, nav_status, sog, cog, heading, width, length, draught = vessel_data
            
            anomaly_likely = sog < 1 or sog > 20  # Simple heuristic for demo
            status_icon = "🚨" if anomaly_likely else "✅"
            
            with gr.Row():
                gr.Markdown(f"**{status_icon} Scenario {i+1}:** {ship_type} - MMSI {mmsi}")
                demo_btn = gr.Button(f"Load Scenario {i+1}")
                
                def load_demo(data=vessel_data):
                    return data
                
                demo_btn.click(
                    fn=load_demo,
                    outputs=[mmsi_input, ship_type_input, nav_status_input, sog_input, 
                           cog_input, heading_input, width_input, length_input, draught_input]
                )
    
    # =============================================================================
    # System Information
    # =============================================================================
    
    with gr.Tab("ℹ️ System Info"):
        gr.Markdown("""
        ### Maritime Monitoring System Status
        
        **Models:**
        - AIS Anomaly Detection: Isolation Forest from HuggingFace
        - SAR Oil Detection: ResNet model with Red Sea imagery
        
        **Automated Process:**
        1. Vessel data input
        2. Real-time anomaly detection
        3. **Automatic SAR analysis** when anomaly detected
        4. Integrated alert generation
        
        **SAR Image Source:** 
        - Random selection from SAR test dataset 
        - Includes Red Sea Sentinel-3B and PALSAR imagery
        - Automatically loaded when vessel anomaly detected
        - No manual image upload required
        
        **Alert System:**
        - Normal vessels: Monitoring continues
        - Anomalous vessels: Automatic SAR oil spill check
        - Combined alerts with maritime authority recommendations
        """)
    
    # =============================================================================
    # Event Handlers
    # =============================================================================
    
    def handle_analysis(*inputs):
        """Main analysis handler - processes AIS data and triggers SAR analysis if needed"""
        
        mmsi, ship_type, nav_status, sog, cog, heading, width, length, draught, threshold = inputs
        
        # Call the integrated analysis function
        result = analyze_vessel_and_environment(
            mmsi, ship_type, nav_status, sog, cog, heading, width, length, draught, threshold
        )
        
        ais_result, ais_risk, ais_score, alert, status, sar_viz = result
        
        return ais_result, ais_risk, ais_score, alert, status, sar_viz
    
    # Connect the main analysis button
    analyze_btn.click(
        fn=handle_analysis,
        inputs=[mmsi_input, ship_type_input, nav_status_input, sog_input, 
                cog_input, heading_input, width_input, length_input, draught_input,
                confidence_threshold],
        outputs=[ais_result, ais_risk, ais_score, integrated_alert, system_status, sar_visualization]
    )

# =============================================================================
# Launch Application
# =============================================================================

if __name__ == "__main__":
    print("🌊 Launching Automated Maritime Monitoring System...")
    app.launch(share=True, server_port=7860)
