# data/structure.txt - Kaggle Data Folder Structure

```
data/
├── train/
│   ├── input/      ← Original images (put your training pairs here)
│   └── target/      ← Target images (same filenames as input/)
├── eval/
│   └── input/      ← Evaluation images (for monitoring during training)
└── test/
    └── input/     ← Test images (for final testing)
```

## IMPORTANT: File Naming
- All training pairs MUST have identical filenames in input/ and target/
- Example: image1.png must exist in BOTH folders

## Files to Include in Kaggle Dataset:
- data/train/input/*.*
- data/train/target/*.*
- data/eval/input/*.* (optional)
- data/test/input/*.* (optional)

## Model Files (already in repo):
- main.py
- model.py
- utils.py
- requirements.txt
- train_kaggle.sh

## Running on Kaggle:
1. Upload this repo + your data as a Kaggle dataset
2. Or put your data in input folders locally before upload
3. Use: python main.py --phase=train