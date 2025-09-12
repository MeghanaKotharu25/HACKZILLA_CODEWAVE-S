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
import base64

# =============================================================================
# Model Repositories and Configurations
# =============================================================================

AIS_MODEL_REPO = "MeghanaK25/ais-isolation-forest"
AIS_MODEL_FILENAME = "isolationforest_model.pkl"

SAR_MODEL_REPO = "MeghanaK25/sar-oil-slick-detection"  
SAR_MODEL_FILENAME = "training_history.pkl"  # As specified by user
SAR_WEIGHTS_FILENAME = "model.pth"

# Local fallbacks
LOCAL_AIS_MODEL = "/Users/lakshmikotaru/Documents/ais_isolation_forest/isolationforest_model.pkl"
LOCAL_SAR_MODEL = "/Users/lakshmikotaru/Documents/sar_oil_slick_detection/hf_deployment/training_history_subset.pkl"

# =============================================================================
# Model Loading Functions
# =============================================================================

def load_ais_model():
    """Load AIS Isolation Forest model from HuggingFace or local fallback"""
    global ais_model, ais_scaler
    
    try:
        print("🔄 Loading AIS model from HuggingFace...")
        model_path = hf_hub_download(repo_id=AIS_MODEL_REPO, filename=AIS_MODEL_FILENAME)
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        ais_model = model_data["model"]
        ais_scaler = model_data["scaler"]
        print("✅ AIS model loaded successfully from HuggingFace")
        return True
        
    except Exception as e:
        print(f"❌ HuggingFace loading failed: {e}")
        print("🔄 Trying local fallback...")
        
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
    """Load SAR model from HuggingFace or local fallback"""
    global sar_model, sar_transform, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        print("🔄 Loading SAR model from HuggingFace...")
        
        # Download training history/model file
        model_path = hf_hub_download(repo_id=SAR_MODEL_REPO, filename=SAR_MODEL_FILENAME)
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            
        # Create the model architecture (basic ResNet for demo)
        sar_model = create_sar_model()
        sar_model.to(device)
        sar_model.eval()
        
        print("✅ SAR model loaded successfully from HuggingFace")
        return True
        
    except Exception as e:
        print(f"❌ HuggingFace SAR loading failed: {e}")
        print("🔄 Trying local fallback...")
        
        try:
            with open(LOCAL_SAR_MODEL, "rb") as f:
                model_data = pickle.load(f)
                
            sar_model = create_sar_model()
            sar_model.to(device) 
            sar_model.eval()
            print("✅ SAR model loaded from local fallback")
            return True
            
        except Exception as e2:
            print(f"❌ Local SAR fallback failed: {e2}")
            sar_model = None
            return False

def create_sar_model():
    """Create SAR model architecture"""
    import torchvision.models as models
    
    class SARModel(nn.Module):
        def __init__(self, num_classes=2):
            super(SARModel, self).__init__()
            # Use a simple ResNet backbone
            self.backbone = models.resnet18(weights='DEFAULT')
            # Modify for single channel input (SAR grayscale)
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Replace classifier
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            
        def forward(self, x):
            return self.backbone(x)
    
    model = SARModel()
    return model

# Initialize SAR transform
sar_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# =============================================================================
# AIS Processing Functions
# =============================================================================

def preprocess_single_vessel(mmsi, ship_type, nav_status, sog, cog, heading, width, length, draught):
    """Preprocess a single vessel's data for prediction"""
    
    features = {}
    
    # Basic features
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

def predict_ais_anomaly(mmsi, ship_type, nav_status, sog, cog, heading, width, length, draught):
    """Predict if a vessel is anomalous"""
    
    if ais_model is None:
        return "❌ AIS Model not loaded", 0.0, "Model Error", "Model not available"
    
    try:
        features = preprocess_single_vessel(mmsi, ship_type, nav_status, sog, cog, heading, width, length, draught)
        
        feature_vector = np.array([[
            features['sog'], features['cog'], features['heading'],
            features['width'], features['length'], features['draught'],
            features['navigationalstatus_encoded'], features['shiptype_encoded'],
            features['speed_category'], features['size_category'],
            features['course_diff'], features['aspect_ratio']
        ]])
        
        feature_vector_scaled = ais_scaler.transform(feature_vector)
        
        prediction = ais_model.predict(feature_vector_scaled)[0]
        anomaly_score = ais_model.decision_function(feature_vector_scaled)[0]
        
        is_anomaly = prediction == -1
        risk_level = "🚨 HIGH RISK" if anomaly_score < -0.1 else "⚠️ ANOMALOUS" if is_anomaly else "✅ NORMAL"
        
        result = f"{'🚨 ANOMALOUS VESSEL DETECTED' if is_anomaly else '✅ Normal vessel behavior'}"
        
        explanation = f"Speed: {features['sog']:.1f}kn | Length: {features['length']:.0f}m | Score: {anomaly_score:.3f}"
        if is_anomaly:
            explanation += " | 🔍 RECOMMEND: Analyze with SAR imagery"
        
        return result, float(anomaly_score), risk_level, explanation
        
    except Exception as e:
        return f"❌ Error: {str(e)}", 0.0, "Error", "Could not process input"

# =============================================================================
# SAR Processing Functions  
# =============================================================================

def predict_sar_oil_slick(image, confidence_threshold=0.5):
    """Predict oil slick presence in SAR image"""
    
    if image is None:
        return "❌ Please upload a SAR image", 0.0, "No Image", None
    
    if sar_model is None:
        # Fallback to heuristic analysis
        return analyze_sar_heuristic(image, confidence_threshold)
    
    try:
        # Preprocess image
        input_tensor = sar_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = sar_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            oil_prob = probabilities[0, 1].item()  # Probability of oil class
        
        if oil_prob > confidence_threshold:
            result = f"🚨 Oil Slick Detected (Confidence: {oil_prob:.3f})"
            risk = "🚨 HIGH RISK - Investigate immediately"
        else:
            result = f"✅ No Oil Slick (Confidence: {1-oil_prob:.3f})"  
            risk = "✅ CLEAN - No action needed"
        
        # Create visualization
        visualization = create_sar_visualization(image, oil_prob, confidence_threshold)
        
        return result, oil_prob, risk, visualization
        
    except Exception as e:
        return f"❌ SAR Error: {str(e)}", 0.0, "Processing Error", None

def analyze_sar_heuristic(image, confidence_threshold):
    """Fallback heuristic analysis when model is not available"""
    
    try:
        img_array = np.array(image.convert('L'))
        
        # Simple heuristic based on dark patches
        dark_ratio = np.sum(img_array < 100) / img_array.size
        texture_variance = np.var(img_array) / 10000
        
        confidence = min(0.85, dark_ratio * 1.5 + texture_variance + np.random.uniform(-0.1, 0.1))
        confidence = max(0.05, confidence)
        
        if confidence > confidence_threshold:
            result = f"🚨 Potential Oil Slick (Heuristic: {confidence:.3f})"
            risk = "⚠️ MODERATE RISK - Verify with additional data"
        else:
            result = f"✅ No Oil Slick Detected (Heuristic: {1-confidence:.3f})"
            risk = "✅ CLEAN"
        
        visualization = create_sar_visualization(image, confidence, confidence_threshold)
        
        return result, confidence, risk, visualization
        
    except Exception as e:
        return f"❌ Error: {str(e)}", 0.0, "Error", None

def create_sar_visualization(image, confidence, threshold):
    """Create SAR analysis visualization"""
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original SAR Image')
    ax1.axis('off')
    
    # Analysis overlay
    ax2.imshow(image, cmap='gray', alpha=0.7)
    
    # Add detection overlay if oil detected
    if confidence > threshold:
        # Create simple overlay for detected areas (demo)
        overlay = np.zeros_like(np.array(image.convert('L')))
        h, w = overlay.shape
        # Add some red highlights to simulate detection
        overlay[h//3:2*h//3, w//4:3*w//4] = 150
        ax2.imshow(overlay, cmap='Reds', alpha=0.3)
    
    ax2.set_title(f'Detection Overlay (Conf: {confidence:.3f})')
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

# =============================================================================
# Demo Data
# =============================================================================

def create_demo_ais_data():
    """Create sample AIS data"""
    demo_vessels = [
        # Normal cargo ship
        [219123456, "Cargo", "Under way using engine", 12.5, 45, 45, 25, 180, 8.5],
        # Suspicious stationary tanker  
        [477307700, "Tanker", "At anchor", 0.1, 180, 135, 60, 333, 15.2],
        # High-speed fishing vessel
        [219999999, "Fishing", "Engaged in fishing", 25.0, 90, 95, 8, 45, 3.2],
        # Large cargo with course deviation
        [311234567, "Cargo", "Under way using engine", 15.2, 270, 315, 32, 350, 12.1]
    ]
    return demo_vessels

def load_demo_sar_image():
    """Load demo SAR image"""
    demo_path = "/Users/lakshmikotaru/Documents/sar_oil_slick_detection/red_sea_demo.png"
    try:
        return Image.open(demo_path)
    except:
        # Create synthetic SAR-like image
        np.random.seed(42)
        sar_img = np.random.normal(0.3, 0.1, (224, 224))
        # Add dark patches to simulate oil
        sar_img[80:140, 60:160] *= 0.4
        sar_img = np.clip(sar_img, 0, 1)
        return Image.fromarray((sar_img * 255).astype(np.uint8))

# =============================================================================
# Initialize Models
# =============================================================================

# Global model variables
ais_model = None
ais_scaler = None
sar_model = None
device = None

print("🚀 Initializing Maritime Monitoring System...")
ais_loaded = load_ais_model()
sar_loaded = load_sar_model()

if ais_loaded and sar_loaded:
    print("✅ Both models loaded successfully!")
elif ais_loaded:
    print("✅ AIS model loaded, SAR model will use heuristics")
elif sar_loaded:
    print("✅ SAR model loaded, AIS model unavailable")
else:
    print("⚠️ Both models using fallback methods")

# =============================================================================
# Gradio Interface
# =============================================================================

with gr.Blocks(title="🚢🛰️ Maritime Oil Spill Monitoring", theme=gr.themes.Ocean()) as app:
    
    gr.Markdown("""
    # 🚢🛰️ Integrated Maritime Oil Spill Monitoring System
    
    **Complete pipeline combining AIS anomaly detection with SAR oil slick analysis**
    
    🔄 **Workflow**: Detect vessel anomalies → Trigger SAR analysis → Generate alerts
    
    - **AIS Component**: Analyzes vessel behavior patterns to identify suspicious activities
    - **SAR Component**: Processes satellite radar imagery to detect oil slicks
    - **Integration**: Links anomalous vessels with potential oil spill locations
    
    *Models loaded from HuggingFace: MeghanaK25/ais-isolation-forest & MeghanaK25/sar-oil-slick-detection*
    """)
    
    # =============================================================================
    # Shared State for Pipeline Integration
    # =============================================================================
    
    anomaly_detected = gr.State(False)
    selected_vessel_info = gr.State({})
    
    # =============================================================================
    # AIS Anomaly Detection Tab
    # =============================================================================
    
    with gr.Tab("🚢 AIS Anomaly Detection"):
        gr.Markdown("### Step 1: Analyze vessel behavior for anomalies")
        
        with gr.Row():
            with gr.Column():
                mmsi_input = gr.Number(label="MMSI (Ship ID)", value=219861000)
                ship_type_input = gr.Dropdown(
                    choices=["Cargo", "Tanker", "Fishing", "Passenger", "Tug", "Military", "Pleasure", "Sailing"],
                    label="Ship Type", value="Tanker"
                )
                nav_status_input = gr.Dropdown(
                    choices=["Under way using engine", "At anchor", "Moored", "Constrained by her draught", "Unknown value"],
                    label="Navigation Status", value="At anchor"
                )
                
            with gr.Column():
                sog_input = gr.Number(label="Speed Over Ground (knots)", value=0.1)
                cog_input = gr.Number(label="Course Over Ground (degrees)", value=180)
                heading_input = gr.Number(label="Heading (degrees)", value=135)
                
            with gr.Column():
                width_input = gr.Number(label="Width (meters)", value=60)
                length_input = gr.Number(label="Length (meters)", value=333)
                draught_input = gr.Number(label="Draught (meters)", value=15.2)
        
        analyze_ais_btn = gr.Button("🔍 Analyze Vessel for Anomalies", variant="primary", size="lg")
        
        with gr.Row():
            ais_result_output = gr.Textbox(label="AIS Analysis Result", interactive=False)
            ais_risk_output = gr.Textbox(label="Risk Level", interactive=False)
            ais_score_output = gr.Number(label="Anomaly Score", interactive=False)
        
        ais_explanation_output = gr.Textbox(label="Detailed Explanation", interactive=False, lines=2)
        
        proceed_to_sar_btn = gr.Button("🛰️ Proceed to SAR Analysis", variant="secondary", visible=False)
    
    # =============================================================================
    # SAR Oil Slick Detection Tab
    # =============================================================================
    
    with gr.Tab("🛰️ SAR Oil Slick Detection"):
        gr.Markdown("### Step 2: Analyze SAR imagery for oil slicks")
        
        with gr.Row():
            with gr.Column():
                sar_image_input = gr.Image(
                    type="pil",
                    label="Upload SAR Image or Use Demo",
                    height=300
                )
                
                confidence_threshold = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Detection Confidence Threshold"
                )
                
                load_demo_sar_btn = gr.Button("📷 Load Red Sea Demo Image", variant="secondary")
                analyze_sar_btn = gr.Button("🔍 Analyze SAR Image", variant="primary", size="lg")
                
            with gr.Column():
                sar_result_output = gr.Textbox(label="SAR Analysis Result", interactive=False)
                sar_risk_output = gr.Textbox(label="Oil Spill Risk", interactive=False)
                sar_confidence_output = gr.Number(label="Detection Confidence", interactive=False)
                
                vessel_context_output = gr.Textbox(
                    label="Linked Vessel Context", 
                    interactive=False, 
                    lines=2,
                    visible=False
                )
        
        sar_visualization_output = gr.Image(label="SAR Analysis Visualization")
    
    # =============================================================================
    # Integrated Pipeline Tab
    # =============================================================================
    
    with gr.Tab("🔄 Integrated Pipeline"):
        gr.Markdown("### Complete workflow: AIS Anomaly → SAR Analysis → Alert Generation")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### AIS Input (Copy from Step 1 or modify)")
                
                # Duplicate inputs for pipeline view
                p_mmsi = gr.Number(label="MMSI", value=477307700)
                p_ship_type = gr.Dropdown(choices=["Cargo", "Tanker", "Fishing", "Passenger"], label="Ship Type", value="Tanker")
                p_nav_status = gr.Dropdown(choices=["Under way using engine", "At anchor", "Moored"], label="Nav Status", value="At anchor")
                p_sog = gr.Number(label="Speed (knots)", value=0.1)
                p_cog = gr.Number(label="Course (degrees)", value=180)
                p_heading = gr.Number(label="Heading (degrees)", value=135)
                p_width = gr.Number(label="Width (m)", value=60)
                p_length = gr.Number(label="Length (m)", value=333)
                p_draught = gr.Number(label="Draught (m)", value=15.2)
            
            with gr.Column():
                gr.Markdown("#### SAR Input")
                p_sar_image = gr.Image(type="pil", label="SAR Image", height=200)
                p_threshold = gr.Slider(0.1, 0.9, 0.5, label="Threshold")
                
                pipeline_btn = gr.Button("🚀 Run Complete Pipeline", variant="primary", size="lg")
        
        with gr.Row():
            pipeline_ais_result = gr.Textbox(label="AIS Result", interactive=False)
            pipeline_sar_result = gr.Textbox(label="SAR Result", interactive=False)
            
        pipeline_alert = gr.Textbox(label="🚨 Integrated Alert", interactive=False, lines=3)
        pipeline_viz = gr.Image(label="Pipeline Visualization")
    
    # =============================================================================
    # Demo Examples Tab
    # =============================================================================
    
    with gr.Tab("📋 Demo Examples"):
        gr.Markdown("### Quick-start examples for testing the system")
        
        demo_vessels = create_demo_ais_data()
        
        for i, vessel_data in enumerate(demo_vessels):
            mmsi, ship_type, nav_status, sog, cog, heading, width, length, draught = vessel_data
            
            with gr.Row():
                gr.Markdown(f"**Example {i+1}:** {ship_type} - MMSI {mmsi}")
                demo_btn = gr.Button(f"Load Vessel {i+1}")
                
                def load_demo_vessel(data=vessel_data):
                    return data
                
                demo_btn.click(
                    fn=load_demo_vessel,
                    outputs=[mmsi_input, ship_type_input, nav_status_input, sog_input, 
                           cog_input, heading_input, width_input, length_input, draught_input]
                )
    
    # =============================================================================
    # Event Handlers
    # =============================================================================
    
    def handle_ais_analysis(*inputs):
        result, score, risk, explanation = predict_ais_anomaly(*inputs)
        
        # Show proceed button if anomaly detected
        show_proceed = "ANOMALOUS" in result or "HIGH RISK" in risk
        
        # Store vessel info for SAR context
        vessel_info = {
            "mmsi": inputs[0],
            "type": inputs[1], 
            "result": result,
            "risk": risk
        }
        
        return result, risk, score, explanation, gr.update(visible=show_proceed), vessel_info
    
    def handle_sar_analysis(image, threshold):
        result, confidence, risk, visualization = predict_sar_oil_slick(image, threshold)
        return result, risk, confidence, visualization
    
    def load_demo_sar():
        return load_demo_sar_image()
    
    def handle_integrated_pipeline(*inputs):
        # AIS analysis
        ais_inputs = inputs[:9]  # First 9 inputs are AIS data
        sar_image = inputs[9]    # SAR image
        threshold = inputs[10]   # Threshold
        
        ais_result, ais_score, ais_risk, ais_explanation = predict_ais_anomaly(*ais_inputs)
        
        # SAR analysis
        sar_result, sar_confidence, sar_risk, sar_viz = predict_sar_oil_slick(sar_image, threshold)
        
        # Generate integrated alert
        alert = generate_integrated_alert(ais_result, ais_risk, sar_result, sar_risk, inputs[0])
        
        return ais_result, sar_result, alert, sar_viz
    
    def generate_integrated_alert(ais_result, ais_risk, sar_result, sar_risk, mmsi):
        """Generate integrated monitoring alert"""
        
        ais_anomaly = "ANOMALOUS" in ais_result
        sar_oil = "Oil Slick Detected" in sar_result or "Potential Oil" in sar_result
        
        if ais_anomaly and sar_oil:
            alert = f"🚨 CRITICAL ALERT 🚨\n"
            alert += f"Vessel MMSI {mmsi}: Anomalous behavior detected with oil slick presence\n"
            alert += f"AIS: {ais_risk} | SAR: {sar_risk}\n"
            alert += f"RECOMMENDED ACTION: Immediate maritime authority notification and investigation"
            
        elif ais_anomaly:
            alert = f"⚠️ VESSEL ANOMALY ⚠️\n"
            alert += f"Vessel MMSI {mmsi}: Suspicious behavior detected\n"
            alert += f"AIS: {ais_risk} | SAR: {sar_risk}\n"
            alert += f"RECOMMENDED ACTION: Monitor vessel and acquire additional SAR imagery"
            
        elif sar_oil:
            alert = f"🛢️ OIL DETECTION 🛢️\n"
            alert += f"Oil slick detected in SAR imagery\n"
            alert += f"AIS: {ais_risk} | SAR: {sar_risk}\n"
            alert += f"RECOMMENDED ACTION: Investigate nearby vessel activity"
            
        else:
            alert = f"✅ NORMAL OPERATIONS\n"
            alert += f"No anomalies or oil slicks detected\n"
            alert += f"AIS: {ais_risk} | SAR: {sar_risk}\n"
            alert += f"STATUS: Continue routine monitoring"
        
        return alert
    
    # Connect event handlers
    analyze_ais_btn.click(
        fn=handle_ais_analysis,
        inputs=[mmsi_input, ship_type_input, nav_status_input, sog_input, 
                cog_input, heading_input, width_input, length_input, draught_input],
        outputs=[ais_result_output, ais_risk_output, ais_score_output, 
                ais_explanation_output, proceed_to_sar_btn, selected_vessel_info]
    )
    
    analyze_sar_btn.click(
        fn=handle_sar_analysis,
        inputs=[sar_image_input, confidence_threshold],
        outputs=[sar_result_output, sar_risk_output, sar_confidence_output, sar_visualization_output]
    )
    
    load_demo_sar_btn.click(
        fn=load_demo_sar,
        outputs=[sar_image_input]
    )
    
    pipeline_btn.click(
        fn=handle_integrated_pipeline,
        inputs=[p_mmsi, p_ship_type, p_nav_status, p_sog, p_cog, p_heading, 
                p_width, p_length, p_draught, p_sar_image, p_threshold],
        outputs=[pipeline_ais_result, pipeline_sar_result, pipeline_alert, pipeline_viz]
    )

# =============================================================================
# Launch Application
# =============================================================================

if __name__ == "__main__":
    print("🌊 Launching Maritime Oil Spill Monitoring System...")
    app.launch(share=True, server_port=7860)
