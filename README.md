Got you! Here's a clean, GitHub-friendly version of your README that keeps all the key info but is more concise and scannable:

---

# Handwritten Amharic OCR with CRNN

A **handwritten Amharic text recognition** system using a **CNN + BiLSTM architecture** with **CTC loss**. Reads cropped images of handwritten Amharic characters and sequences.

## Features

* Preprocesses images: grayscale conversion, height normalization, and pixel normalization
* **CRNN** for variable-length sequence recognition
* Trained with **CTC loss** (alignment-free sequence prediction)
* Supports train/validation datasets

## Dataset

The dataset is **not included** due to size. Contact for access:

* [rafiakedir22@gmail.com](mailto:rafiakedir22@gmail.com)
* [danliliyah5@gmail.com](mailto:danliliyah5@gmail.com)

## Usage

1. Clone the repo:

```bash
git clone <repo-url>
cd <repo-folder>
```

2. Install dependencies:

```bash
pip install torch torchvision pillow
```

3. Prepare dataset structure:

```
Formatted_Data/
├─ train/
│  ├─ cropped_images/
│  └─ labels/
└─ val/
   ├─ cropped_images/
   └─ labels/
```

4. Run training:

```bash
python train.py
```

> Adjust `batch_size` and `learning_rate` as needed. Training time depends on dataset size and GPU availability.

## Notes

* Load `amharic_mapping.py` before training
* Training may be slow for large datasets; reduce batch size if needed

## Contact

For questions, dataset access, or collaboration, email:

* [rafiakedir22@gmail.com](mailto:rafiakedir22@gmail.com)
* [danliliyah5@gmail.com](mailto:danliliyah5@gmail.com)
