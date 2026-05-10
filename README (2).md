# 🦺 PPE Detection System using YOLOv8

## 📌 Project Overview
This project is a Real-Time PPE (Personal Protective Equipment) Detection System developed using YOLOv8 and Streamlit.  
The system detects construction workers and identifies whether they are wearing proper safety equipment such as:
- Hardhat
- Safety Vest
- Mask

It also detects PPE violations and displays worker compliance analysis in real time.

---

## 🚀 Features
- Real-time PPE Detection
- PPE Violation Identification
- YOLOv8 Custom Trained Model
- Streamlit Web Interface
- Image Upload Support
- Video Upload Support
- Webcam Detection
- Per-Worker Compliance Analysis
- Detection Result Download

---

## 🧠 Technologies Used
- Python
- YOLOv8
- OpenCV
- Streamlit
- NumPy
- Pandas
- Google Colab

---

## 📂 Dataset
Dataset used:
Construction Site Safety Dataset from Roboflow

Classes:
- Hardhat
- Safety Vest
- Person
- No-Hardhat
- No-Safety-Vest
- Mask
- Machinery
- Vehicle
- Safety Cone

---

## 📊 Model Performance
| Parameter | Result |
|---|---|
| Model | YOLOv8n |
| mAP@0.5 | ~72% |
| Precision | ~85% |
| Recall | ~64% |
| Inference Speed | ~268ms |

---

## ⚙️ Project Workflow
1. Dataset Collection
2. Data Preprocessing
3. YOLOv8 Model Training
4. Model Evaluation
5. Streamlit Integration
6. Real-Time Detection
7. PPE Compliance Monitoring

---

## ▶️ Run Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
python -m streamlit run app.py
```

---

## 📁 Project Structure

```text
PPE-Detection/
│
├── app.py
├── ppe_best.pt
├── requirements.txt
├── README.md
├── data.yaml
└── sample_images/
```

---

## ✅ Project Outcome
The system successfully detects PPE compliance violations in construction environments and provides real-time monitoring support for workplace safety.

---

## 👨‍💻 Developer
Sai Raghuveer Reddy

Internship Major Project – 2026
