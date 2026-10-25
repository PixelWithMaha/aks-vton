# AKS: Virtual Try-On for South Asian Garments

**FAST-NUCES + LAAM Collaboration**

## Overview
Virtual try-on system for South Asian multi-piece garments (kameez, lehenga, saree, gharara, etc.). Users upload a photo and see themselves in LAAM products.

## Team
- **Maheen Kamran Zeb** 
- **Maaham Rizwan** 
- **Izma Asif** 
- **Supervisor:** Ms Umm-e-Ammarah

## Timeline
- Weeks 1-4: Dataset cleaning
- Weeks 5-6: Preprocessing
- Weeks 7-10: Fine-tune IDM-VTON
- Weeks 11-12: FastAPI + evaluation

## Quick Start
```bash
git clone (https://github.com/PixelWithMaha/Aks-By-LAAM-Our-Virtual-Try-On)
cd Aks-By-LAAM-Our-Virtual-Try-On
dvc pull  # Get data from DagsHub
python scripts/01_filter/smart_filter.py
```

## Repository Structure

aks-dataset/
├── scripts/
│ ├── 01_filter/ # FiftyOne auto-filtering
│ ├── 02_preprocess/ # MediaPipe, DensePose, SAM2, SCHP
│ └── 03_train/ # IDM-VTON fine-tuning
├── data/
│ ├── raw/ # Original 9,456 images (DVC tracked)
│ ├── cleaned/v2.0/ # After filtering
│ └── preprocessed/ # After pose/mask extraction
├── reports/ # Metrics, plots, logs
├── notebooks/ # Exploratory analysis
├── annotations/ # Label Studio exports
├── dvc.yaml # Pipeline definition
├── params.yaml # Hyperparameters
└── requirements.txt # Dependencies

## Tools Used
- **FiftyOne:** Dataset curation
- **FastDup + DINOv2:** Semantic deduplication
- **BRISQUE + Laplacian:** Image quality
- **MediaPipe:** Pose detection
- **DagsHub:** Data versioning
- **IDM-VTON:** Base try-on model
