# Adversarial Patch Pipeline

A comprehensive Python-based proof-of-concept (PoC) for generating adversarial patches capable of bypassing multiple, robust, real-world computer vision pipeline defenses.

## Overview

This project implements a full-featured adversarial patch system that:

- **Generates adversarial patches** optimized to bypass multiple defense layers
- **Tests against multiple models** (YOLOv5, ResNet, EfficientNet)
- **Implements 5 defense layers**: Input normalization, adversarial detection, multi-frame smoothing, context rules, and model ensemble
- **Provides comprehensive evaluation** with metrics, visualizations, and reports
- **Includes real-world deployment** scripts for physical patch testing

## Features

### Attack Capabilities
- Universal and targeted adversarial patch generation
- Multi-model attack (YOLOv5, ResNet, EfficientNet)
- Defense-aware optimization (bypasses all 5 defense layers)
- Physical-world considerations (lighting, angles, transformations)

### Defense Layers
1. **Input Normalization**: Random brightness, blur, affine transforms, noise
2. **Adversarial Detection**: Norm-based + learned OOD detector
3. **Multi-Frame Smoothing**: Consensus across consecutive frames
4. **Context Rules**: Temporal, spatial, and environmental validation
5. **Model Ensemble**: Consensus-based detection across multiple models

### Evaluation & Analysis
- Comprehensive metrics (success rates, bypass rates, defense effectiveness)
- Scenario-based evaluation (lighting, angles, camera conditions)
- Frame-by-frame analysis
- Visualizations and detailed reports

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ASK92/Adverserial-AI/
cd adversarial-patch-pipeline

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Generate Configuration

```bash
python main.py init-config --output config.yaml
```

#### 2. Train an Adversarial Patch

```bash
python main.py train-patch \
    --patch-size 100 100 \
    --iterations 1000 \
    --learning-rate 0.1 \
    --output data/patches/patch.pt
```

#### 3. Evaluate Patch

```bash
python main.py evaluate \
    --patch-path data/patches/patch.pt \
    --output-dir data/results
```

### Using Jupyter Notebooks

1. **Patch Training**: `notebooks/01_patch_training.ipynb`
2. **Pipeline Demo**: `notebooks/02_pipeline_demo.ipynb`
3. **Ablation Study**: `notebooks/03_ablation_study.ipynb`

## 📁 Project Structure

```
adversarial-patch-pipeline/
├── src/
│   ├── models/          # Model loaders (YOLOv5, ResNet, EfficientNet)
│   ├── defenses/        # Defense layer implementations
│   ├── patch/           # Patch generation and optimization
│   ├── evaluation/      # Evaluation framework
│   └── utils/           # Utilities (config, logging, visualization)
├── notebooks/           # Jupyter notebooks for demos
├── scripts/             # Deployment and utility scripts
├── data/
│   ├── models/          # Model checkpoints
│   ├── patches/         # Trained patches
│   └── results/         # Evaluation results
├── main.py              # CLI entry point
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Configuration

The system uses YAML configuration files. Generate a default config:

```bash
python main.py init-config
```

Key configuration sections:

- **Models**: Model selection and parameters
- **Defenses**: Defense layer configurations
- **Patch**: Patch generation parameters
- **Evaluation**: Evaluation settings

## Evaluation Metrics

The evaluation framework computes:

- **Attack Metrics**:
  - Single model success rates
  - Ensemble success rate
  - Defense bypass rate
  - Scenario-specific success rates
  - Frame-by-frame consistency

- **Defense Metrics**:
  - Defense effectiveness
  - False negative rate
  - Defense layer breakdown

## Defense Layers

### 1. Input Normalization
Random transformations to reduce patch effectiveness:
- Brightness adjustment (0.7-1.3x)
- Gaussian blur (30% probability)
- Affine transformations (rotation, translation, scale)
- Gaussian noise

### 2. Adversarial Detection
Detects adversarial examples:
- L2 norm-based detection
- Learned out-of-distribution detector
- Combined consensus

### 3. Multi-Frame Smoothing
Requires consensus across frames:
- Window size: 5 frames
- Consensus threshold: 60%
- Temporal consistency check

### 4. Context Rules
Validates detections using context:
- Temporal consistency
- Spatial consistency
- Environmental cues (lighting, time of day)

### 5. Model Ensemble
Consensus across multiple models:
- Multiple architecture agreement
- Configurable consensus threshold

## Adversarial Patch Generation

The patch optimizer uses a multi-component loss function:

- **Classification Loss**: Attack model predictions
- **Detection Loss**: Attack object detection models
- **Defense Evasion Loss**: Bypass defense layers
- **Total Variation Loss**: Patch smoothness

## Results & Reports

Evaluation generates:

- **Text Report**: Comprehensive metrics and analysis
- **CSV Summary**: Machine-readable metrics
- **Visualizations**: Charts and graphs
- **Frame Analysis**: Per-frame detection results

## Research & References

This implementation is based on:

- **REAP**: Robust Physical-Patch Benchmark (ICCV 2023)
- **NeurIPS 2024**: Defense-Evading Patch Methods
- **Adversarial Robustness Toolbox (ART)**
- **Foolbox**: Adversarial attacks library

## Real-World Deployment

### Physical Patch Deployment

1. **Print Patch**:
```bash
python scripts/deploy_patch.py \
    --patch data/patches/patch.pt \
    --mode print \
    --output patch_printable.png \
    --size 10 10 \
    --dpi 300
```

2. **Capture Video**:
```bash
python scripts/deploy_patch.py \
    --patch data/patches/patch.pt \
    --mode capture \
    --output captured_video.mp4 \
    --duration 10
```

3. **Process Video**:
```bash
python scripts/process_video.py \
    --video captured_video.mp4 \
    --output processed_output.mp4
```

## Ablation Studies

Run ablation studies to evaluate individual defense layers:

```bash
jupyter notebook notebooks/03_ablation_study.ipynb
```

This evaluates:
- Each defense layer individually
- Defense combinations
- Effectiveness analysis

## Extending the Framework

### Adding New Models

```python
from src.models.model_loader import ModelLoader

loader = ModelLoader(device='cuda')
model = loader.load_resnet('resnet101', pretrained=True)
```

### Adding Custom Defenses

```python
from src.defenses.defense_pipeline import DefensePipeline

class CustomDefense:
    def __call__(self, image):
        # Your defense logic
        return processed_image

defense_pipeline = DefensePipeline(
    custom_defense=CustomDefense(),
    enabled=True
)
```

### Custom Loss Functions

Modify `src/patch/patch_optimizer.py` to add custom loss components.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use CPU: `--device cpu`

2. **Model Loading Fails**:
   - Check internet connection (for pretrained weights)
   - Verify model names in config

3. **Import Errors**:
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+)

## 📄 License

All dependencies are openly available. See `requirements.txt` for full list.

## Contributing

This is a research PoC. For extensions:

1. Follow modular architecture
2. Add comprehensive docstrings
3. Include unit tests
4. Update documentation

## Documentation

- **API Documentation**: See docstrings in source code
- **Notebooks**: Interactive examples in `notebooks/`
- **Scripts**: Utility scripts in `scripts/`

## Educational Use

This project is designed for:
- Research on adversarial robustness
- Security testing of CV systems
- Educational demonstrations
- Defense mechanism evaluation

## Disclaimer

This tool is for **research and educational purposes only**. Use responsibly and ethically. Do not use for malicious purposes.

## Related Work

- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [Foolbox](https://github.com/bethgelab/foolbox)
- [REAP Benchmark](https://github.com/REAP-benchmark)

---

**Version**: 1.0.0  
**Last Updated**: 2025

