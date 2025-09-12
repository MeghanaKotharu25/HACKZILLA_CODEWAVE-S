# 🚢🛰️ Maritime Oil Spill Monitoring System

**WORKING PROTOTYPE**: A complete integrated pipeline combining **AIS (Automatic Identification System) anomaly detection** with **SAR (Synthetic Aperture Radar) oil slick detection** for maritime environmental monitoring.

✅ **Status**: FULLY FUNCTIONAL with models loaded from HuggingFace

## 🌊 Overview

This pipeline integrates two AI models:
1. **AIS Isolation Forest Model**: Detects anomalous vessel behavior from AIS data
2. **SAR ResNet Model**: Identifies oil slicks in Sentinel-1 radar imagery

When anomalous vessel activity is detected, the system analyzes corresponding SAR imagery to check for potential oil spills in the same area.

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Integrated Maritime Monitoring System

**Primary Method** (Recommended):
```bash
# Use the launcher script
./run_app.sh
```

**Direct Method**:
```bash
# Run the merged Gradio application
python merged_app.py
```

The app will be available at `http://localhost:7860` with a public sharing URL

**Legacy Streamlit** (Alternative):
```bash
streamlit run streamlit_app.py
```

### Testing Models

Before running the full app, test model loading:

```bash
python test_models.py
```

## 📁 Files Structure

```
/Users/lakshmikotaru/Documents/sar_oil_slick_detection/
├── merged_app.py              # ✨ MAIN APPLICATION - Integrated Gradio interface
├── run_app.sh                # 🚀 Launcher script (recommended)
├── requirements_merged.txt    # Dependencies for merged app
├── streamlit_app.py           # Legacy Streamlit dashboard
├── test_models.py             # Model loading verification script
├── requirements.txt           # Python dependencies (legacy)
├── demo_ais_data.csv         # Sample AIS data for testing
├── red_sea_demo.png          # Demo SAR image with simulated oil slicks
├── create_demo_image.py      # Script to generate demo SAR images
└── README.md                 # This file
```

## 🤖 Model Details

### AIS Anomaly Detection Model
- **Source**: Hugging Face repository `MeghanaK25/ais-isolation-forest`
- **Local Fallback**: `/Users/lakshmikotaru/Documents/ais_isolation_forest/isolationforest_model.pkl`
- **Type**: Scikit-learn Isolation Forest
- **Purpose**: Detects unusual vessel movement patterns, speeds, or behaviors

### SAR Oil Slick Detection Model
- **Source**: Hugging Face repository `MeghanaK25/sar-oil-slick-detection` (with local fallback)
- **Local Fallback**: `/Users/lakshmikotaru/Documents/sar_oil_slick_detection/hf_deployment/oil_slick_resnet_subset.pth`
- **Architecture**: ResNet-50 based classification model adapted for single-channel SAR imagery
- **Purpose**: Identifies oil slick presence in SAR imagery with confidence scoring

## 🔄 Pipeline Workflow

1. **AIS Data Upload**: Upload CSV file containing vessel tracking data
2. **Anomaly Detection**: Run Isolation Forest model to identify suspicious vessels
3. **Anomaly Selection**: Choose specific anomalous events for investigation
4. **SAR Image Analysis**: Upload SAR imagery or use Red Sea demo image
5. **Oil Slick Detection**: Process SAR image to detect potential oil spills
6. **Risk Assessment**: Get probability maps, coverage percentages, and risk levels

## 📊 Dashboard Features

### Left Panel - AIS Anomaly Detection
- CSV file upload for AIS data
- Real-time anomaly detection using Isolation Forest
- Interactive anomaly selection and inspection
- Demo mode with sample data

### Right Panel - SAR Oil Slick Detection
- SAR image upload (PNG/JPG)
- Adjustable detection threshold (0.1-0.9)
- Red Sea demo image integration
- Real-time oil slick analysis

### Results Display
- **Original SAR Image**: Input imagery
- **Probability Map**: Pixel-level oil detection confidence
- **Detection Mask**: Binary mask overlay showing detected regions
- **Risk Metrics**: Coverage percentage, max/mean confidence, risk level

## 🔧 Technical Implementation

### Model Loading Strategy
1. **Primary**: Load models from Hugging Face repositories
2. **Fallback**: Use local model files if HF is unavailable
3. **Weight Adaptation**: Custom handling for architecture mismatches (e.g., 3→1 channel conversion)

### SAR Image Processing
- Grayscale conversion for single-channel SAR
- Resize to 224×224 for model compatibility
- Normalization with mean=0.5, std=0.5
- Tensor preprocessing pipeline

### AIS Data Processing
- Automatic numeric feature extraction
- Missing value imputation with median values
- Isolation Forest scoring and anomaly labeling

## 📈 Performance Metrics

### Model Statistics
- **AIS Model**: Loaded from Hugging Face successfully
- **SAR Model**: 23.5M parameters, ResNet-50 based architecture
- **Loading Time**: ~15-30 seconds (includes model download/loading)
- **Inference Time**: <2 seconds per SAR image

### Detection Capabilities
- **Oil Coverage**: Percentage of image area with detected oil
- **Confidence Scoring**: Pixel-level probability maps (0-1)
- **Risk Assessment**: Automated classification (CLEAN/LOW/MODERATE/HIGH)

## 🌍 Use Cases

### Maritime Environmental Monitoring
- Real-time oil spill detection from satellite imagery
- Vessel behavior analysis for compliance monitoring
- Integration with maritime traffic control systems

### Research Applications
- Marine pollution studies
- AIS data pattern analysis
- SAR image processing pipeline development

### Operational Deployment
- Coast guard monitoring systems
- Environmental protection agencies
- Marine insurance assessments

## ⚡ Performance Optimization

The pipeline is optimized for speed and reliability:
- **Caching**: Streamlit caching for model loading
- **Batch Processing**: Efficient tensor operations
- **Memory Management**: Optimized for single-image inference
- **Error Handling**: Robust fallback mechanisms

## 🔍 Demo Data

### Sample AIS Data (`demo_ais_data.csv`)
- 20 vessel records with position, speed, course data
- Mix of normal and anomalous patterns
- Red Sea maritime area focus

### Demo SAR Image (`red_sea_demo.png`)
- Synthetic SAR-like imagery (224×224)
- Simulated oil slick patterns
- Speckle noise and texture typical of SAR

## 📝 Usage Examples

### Basic Usage
1. Start the Streamlit app
2. Enable "Use demo" mode
3. Click "Analyze SAR image" to see oil detection results

### Custom Data Analysis
1. Upload your AIS CSV file
2. Run anomaly detection
3. Upload corresponding SAR imagery
4. Adjust detection threshold as needed
5. Analyze results and risk assessment

## 🛠️ Troubleshooting

### Model Loading Issues
- Check internet connection for Hugging Face access
- Verify local model files exist at specified paths
- Review error messages for specific loading problems

### SAR Image Processing
- Ensure images are in supported formats (PNG, JPG)
- Check image dimensions and color channels
- Verify SAR imagery quality and contrast

### Performance Issues
- Close other applications to free memory
- Consider reducing image resolution for faster processing
- Check available disk space for model caching

## 🔗 Integration

The pipeline is designed for integration with:
- **Airflow**: Automated scheduling and orchestration
- **Sentinel-1 API**: Direct SAR imagery access
- **AIS data streams**: Real-time vessel tracking integration
- **Alert systems**: Automated notification for oil detection

## 📧 Support

For technical support or questions about the pipeline:
- Review error logs in the terminal/console
- Check model loading with `test_models.py`
- Verify data format compatibility
- Ensure all dependencies are properly installed

---

**🎯 Ready to detect oil spills with AI-powered maritime monitoring!** 

This pipeline demonstrates the power of combining multiple data sources and AI models for environmental protection and maritime safety.
