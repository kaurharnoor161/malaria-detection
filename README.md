# 🦠 Malaria Detection System using Deep Learning

A deep learning-based malaria cell classification system built using TensorFlow, MobileNetV2, and Streamlit.

This project detects whether a blood smear cell image is:

- ✅ Uninfected
- ⚠️ Parasitized

using transfer learning and computer vision.

---

# 🚀 Features

- Deep Learning based medical image classification
- Transfer Learning using MobileNetV2
- TensorFlow + Keras implementation
- Streamlit web application
- Real-time image prediction
- Fine-tuned model for high accuracy
- Automated training visualization
- Early stopping to prevent overfitting

---

# 📊 Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | 95.67% |
| Architecture | MobileNetV2 |
| Dataset Size | 27,558 Images |
| Classes | 2 |

---

# 🧠 Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Streamlit
- Pillow
- Scikit-learn

---

# 📁 Project Structure

```bash
malaria_detection/
│
├── dataset/
│   ├── Parasitized/
│   └── Uninfected/
│
├── model/
│   └── malaria_mobilenetv2.keras
│
├── train.py
├── app.py
├── requirements.txt
└── README.md