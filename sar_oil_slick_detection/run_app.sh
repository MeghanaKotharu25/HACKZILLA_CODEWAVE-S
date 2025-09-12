#!/bin/bash

echo "🌊 Starting Maritime Oil Spill Monitoring System"
echo "================================================"

# Check if required dependencies are installed
echo "Checking dependencies..."

if ! python -c "import gradio" 2>/dev/null; then
    echo "❌ Gradio not found. Installing dependencies..."
    pip install -r requirements_merged.txt
fi

# Create the Red Sea demo image if it doesn't exist
if [ ! -f "red_sea_demo.png" ]; then
    echo "Creating Red Sea demo image..."
    python create_demo_image.py
fi

echo "🚀 Launching application..."
echo "The app will be available at: http://localhost:7860"
echo "A public URL will also be generated for sharing"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Launch the corrected automated maritime monitoring application
python maritime_monitoring_app.py
