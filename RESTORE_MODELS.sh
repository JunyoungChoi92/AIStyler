#!/bin/bash
# Script to restore model files from split archives

echo "Restoring model files from split archives..."

# Restore checkpoint_2c_466
if [ -f "checkpoint_2c_466/checkpoint_2c_466_models.tar.gz.aa" ]; then
    echo "Restoring checkpoint_2c_466 models..."
    cd checkpoint_2c_466
    cat checkpoint_2c_466_models.tar.gz.* > checkpoint_2c_466_models.tar.gz
    tar -xzf checkpoint_2c_466_models.tar.gz
    rm checkpoint_2c_466_models.tar.gz
    cd ..
    echo "✓ checkpoint_2c_466 restored"
else
    echo "⚠ checkpoint_2c_466 split files not found"
fi

# Restore correction_checkpoint_3200
if [ -f "correction_checkpoint_3200/correction_checkpoint_3200_models.tar.gz.aa" ]; then
    echo "Restoring correction_checkpoint_3200 models..."
    cd correction_checkpoint_3200
    cat correction_checkpoint_3200_models.tar.gz.* > correction_checkpoint_3200_models.tar.gz
    tar -xzf correction_checkpoint_3200_models.tar.gz
    rm correction_checkpoint_3200_models.tar.gz
    cd ..
    echo "✓ correction_checkpoint_3200 restored"
else
    echo "⚠ correction_checkpoint_3200 split files not found"
fi

# Restore detection_checkpoint_3300
if [ -f "detection_checkpoint_3300/detection_checkpoint_3300_models.tar.gz.aa" ]; then
    echo "Restoring detection_checkpoint_3300 models..."
    cd detection_checkpoint_3300
    cat detection_checkpoint_3300_models.tar.gz.* > detection_checkpoint_3300_models.tar.gz
    tar -xzf detection_checkpoint_3300_models.tar.gz
    rm detection_checkpoint_3300_models.tar.gz
    cd ..
    echo "✓ detection_checkpoint_3300 restored"
else
    echo "⚠ detection_checkpoint_3300 split files not found"
fi

echo "Model restoration complete!"
echo ""
echo "You should now have the following model files:"
echo "- checkpoint_2c_466/adapter_model.safetensors"
echo "- checkpoint_2c_466/optimizer.pt"
echo "- correction_checkpoint_3200/adapter_model.safetensors"
echo "- detection_checkpoint_3300/adapter_model.safetensors"
echo "And other associated .pt, .pth, .bin files"