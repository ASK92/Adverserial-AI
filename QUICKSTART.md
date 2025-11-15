# Quick Start Guide

Get up and running with the adversarial patch pipeline in 5 minutes.

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Generate Config

```bash
python main.py init-config
```

## 3. Train a Patch (Quick Demo)

```bash
python main.py train-patch \
    --patch-size 100 100 \
    --iterations 500 \
    --output data/patches/demo_patch.pt
```

## 4. Evaluate

```bash
python main.py evaluate \
    --patch-path data/patches/demo_patch.pt \
    --output-dir data/results
```

## 5. View Results

Check `data/results/` for:
- Evaluation report (`.txt`)
- CSV summary (`.csv`)
- Visualizations (`.png`)

## Jupyter Notebooks

For interactive exploration:

```bash
jupyter notebook notebooks/01_patch_training.ipynb
```

## Next Steps

- Read `README.md` for full documentation
- Explore `notebooks/` for detailed examples
- Check `scripts/` for deployment utilities


