<div align="center">

# 🦟 Malaria Detection System

### Teaching a neural network to spot a parasite that microscopists have hunted by eye for 140 years

**97.2% Test Accuracy** · **97.5% Sensitivity** · **MobileNetV2 + Transfer Learning**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

</div>

---

## The Problem

Every year, malaria infects roughly **247 million people** and kills more than **619,000** — most of them in Sub-Saharan Africa and South Asia, and disproportionately children. The diagnostic gold standard hasn't changed in over a century: a technician stains a blood smear with Giemsa dye, puts it under a microscope, and counts infected cells by eye.

That works — but it's slow (20–45 minutes per sample), it demands scarce expertise, and it's human, which means fatigue and workload push misdiagnosis rates as high as **30%** in the field. Rapid diagnostic tests help, but they miss low-parasitaemia infections that a trained eye — or a well-trained model — would catch.

This project asks: *what if a phone-sized neural network could do the counting instead?*

## The Approach

Rather than building a CNN from scratch and hoping it learns what a parasite looks like from 27,558 images, this system stands on the shoulders of **MobileNetV2**, pretrained on 1.4 million ImageNet images. The intuition: a network that already understands edges, textures, and shapes shouldn't have to relearn vision from zero — it just needs to be taught what malaria *specifically* looks like.

Training happens in two deliberate phases:

| Phase | What's frozen | What's learning | Learning rate |
|---|---|---|---|
| **1 — Head training** | Entire MobileNetV2 backbone | Custom classification head (~363K params) | `1e-4` |
| **2 — Fine-tuning** | Bottom layers of the backbone | Top 40 backbone layers + head | `1e-5` (10x lower) |

The lower learning rate in Phase 2 matters more than it looks — crank it up and you get *catastrophic forgetting*, where the pretrained features get overwritten by noisy early gradients before the model has anything useful to replace them with. Going slow lets the network gently nudge its ImageNet knowledge toward parasite-specific patterns instead of erasing it.

## Results

Trained and evaluated on the **NIH Malaria Cell Images Dataset** (27,558 images, perfectly balanced between Parasitized and Uninfected), held out on a 5,511-image test set the model never saw during training:

| Metric | Score | Why it matters |
|---|---|---|
| **Accuracy** | 97.2% | Beats custom CNN baselines by 2+ points |
| **Sensitivity (Recall)** | 97.5% | Clears WHO's 95% threshold for RDT sensitivity — misses fewer real infections |
| **Specificity** | 96.9% | Doesn't cry wolf on healthy cells |
| **F1 Score** | 97.1% | Precision and recall in balance |
| **Parameters** | 3.4M | ~40x smaller than VGG-16, while *outperforming* it |

For context: field microscopists typically operate at **75–90% accuracy**, and that number erodes further under fatigue. A model doesn't get tired at 2am on a double shift.

<details>
<summary><b>How it stacks up against other architectures</b></summary>

| Model | Accuracy | Parameters |
|---|---|---|
| Custom CNN (3 blocks) | 93–95% | ~2M |
| VGG-16 (fine-tuned) | 95–96% | 138M |
| ResNet-50 | 96–97% | 25M |
| **MobileNetV2 (this project)** | **97.2%** | **3.4M** |

</details>

## Try It

Upload a blood cell image, get a verdict and a confidence score, no ML knowledge required.

```bash
git clone https://github.com/kaurharnoor161/malaria-detection.git
cd malaria-detection
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Drop in a cell image and it'll tell you — with a confidence badge ranging from *Low* to *Very High* — whether it's looking at a healthy cell or a parasitized one.

## Train It Yourself

Want to retrain on your own data, or just see the pipeline run end to end?

```bash
python train.py
```

This expects a `dataset/` folder with `Parasitized/` and `Uninfected/` subdirectories (the [NIH dataset on Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) is a drop-in fit). It'll run both training phases, evaluate on a held-out test split, save the model to `model/`, and plot accuracy/loss curves so you can watch it learn.

## Under the Hood

```
malaria_detection/
├── dataset/
│   ├── Parasitized/          # 13,779 infected cell images
│   └── Uninfected/           # 13,779 healthy cell images
├── model/
│   └── malaria_mobilenetv2.keras
├── train.py                  # Two-phase training pipeline
├── app.py                    # Streamlit inference app
└── requirements.txt
```

**Preprocessing pipeline:** every image gets resized to 96×96, normalized, and cast to RGB before it ever reaches the model — training and inference share this exact path, so what the model sees in production matches what it learned on.

**Data augmentation** (training set only): ±20° rotation, horizontal/vertical flips, ±15% zoom, ±10% shifts, brightness jitter — simulating the messiness of real-world slide acquisition so the model doesn't overfit to pristine lab conditions.

**Regularization:** dropout (0.4 → 0.3), batch normalization, and early stopping on validation loss keep the train/validation accuracy gap tight throughout training.

## Where This Doesn't Belong (Yet)

Honesty matters more than hype here. This is a research and educational proof-of-concept, not a diagnostic device:

- Trained and tested on **one** dataset — generalization to different microscopes, stains, and populations is unvalidated.
- **Binary only** — infected vs. uninfected, no species identification (*P. falciparum* vs. *P. vivax*, etc.) or parasitaemia density.
- Works on **pre-segmented single cells**, not raw blood smear slides — there's no cell-detection step yet.
- **No clinical trials, no regulatory review.** It has not been tested against expert diagnoses on real patient samples.

## What's Next

- [ ] Multi-class species identification
- [ ] Whole-slide processing via object detection (YOLO / Faster R-CNN) instead of pre-cropped cells
- [ ] Parasitaemia density estimation
- [ ] Grad-CAM visualizations so clinicians can see *why* the model flagged a cell
- [ ] TensorFlow Lite conversion for offline mobile deployment in the field
- [ ] Multi-site clinical validation
- [ ] Ensemble with EfficientNetB3 / ResNet50

## Tech Stack

`TensorFlow / Keras` · `MobileNetV2` · `Streamlit` · `scikit-learn` · `NumPy` · `Pillow` · `Matplotlib`

## Acknowledgments

Built on the **NIH Malaria Cell Images Dataset**, created by Rajaraman et al. and hosted on Kaggle — the standard benchmark for this problem, and the reason a project like this can exist without collecting a single blood sample.

---

<div align="center">

**⚠️ For research and educational purposes only. Not a substitute for professional medical diagnosis.**

Built by [Harnoor Kaur](https://github.com/kaurharnoor161)

</div>
