#!/bin/bash

# Korea Times Style Correction - Quick Start Script

echo "======================================================================"
echo "Korea Times Style Correction - Quick Start"
echo "Checkpoint: stage_2C_mixed_review/checkpoint-466"
echo "======================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if running in correct directory
if [ ! -f "inference.py" ]; then
    echo "Error: Please run this script from the inference_test_package directory"
    exit 1
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import torch; import transformers; import unsloth" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "Dependencies not found. Installing..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

echo "✓ Dependencies OK"
echo ""

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: CUDA not available. This script requires a GPU."
    echo "Please run on a machine with CUDA-capable GPU."
    exit 1
fi

echo "✓ CUDA OK"
echo ""

# Run example inference
echo "Running example inference on sample_article.txt..."
echo ""

python3 inference.py \
    --checkpoint checkpoint \
    --input examples/sample_article.txt \
    --output examples/test_output.json \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Inference completed successfully!"
    echo "======================================================================"
    echo ""
    echo "Output saved to: examples/test_output.json"
    echo ""
    echo "To view results:"
    echo "  cat examples/test_output.json"
    echo ""
    echo "To run interactive mode:"
    echo "  python3 inference.py --interactive --checkpoint checkpoint"
    echo ""
    echo "To process your own file:"
    echo "  python3 inference.py --checkpoint checkpoint --input your_file.txt --output result.json"
    echo ""
else
    echo ""
    echo "Error: Inference failed"
    echo "Check error messages above for details"
    exit 1
fi
