# Large Model Files Required

This repository contains the Korean Times SLM inference system. Due to GitHub's file size limitations, the following large model files need to be downloaded separately:

## Required Model Files

### checkpoint_2c_466/
- `adapter_model.safetensors` (333MB)
- `optimizer.pt` (663MB)
- `rng_state.pth`
- `scheduler.pt`
- `trainer_state.json`
- `training_args.bin`

### correction_checkpoint_3200/
- `adapter_model.safetensors` (167MB)
- `optimizer.pt`
- `rng_state.pth`
- `scheduler.pt`
- `trainer_state.json`
- `training_args.bin`

### detection_checkpoint_3300/
- `adapter_model.safetensors` (167MB)
- `optimizer.pt`
- `rng_state.pth`
- `scheduler.pt`
- `trainer_state.json`
- `training_args.bin`

## How to Obtain Model Files

Please contact the repository owner or check the releases section for download links to these model files.

Once downloaded, place them in their respective checkpoint directories before running the inference scripts.