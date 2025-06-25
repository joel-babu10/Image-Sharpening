# Image Sharpening using Knowledge Distillation

A lightweight, real-time image sharpening system designed for video conferencing scenarios. This project leverages a **Teacher-Student knowledge distillation framework** to transfer knowledge from a high-performing deep model (Restormer) to an efficient student model capable of operating at 30–60 FPS on 1080p images.

---

## 🔍 Project Summary

- **Goal:** Improve visual clarity in video streams suffering from blurriness caused by compression, noise, or low bandwidth.
- **Approach:** Use a deep Restormer-based Teacher model to train a lightweight Student model via knowledge distillation.
- **Output:** A sharp image or video frame with enhanced structural and perceptual quality.

---

## 📐 Architecture

      +---------------------+
      |   Teacher Model     |
      |   (Restormer)       |
      +---------------------+
                ↓
    Feature & Perceptual Loss
                ↓
      +---------------------+
      |   Student Model     |
      |  (UNet/ResUNetLite) |
      +---------------------+
                ↓
     Reconstructed Sharp Image
     
---

## 📦 Features

- ✅ Deep teacher model (Restormer or ResUNet)
- ✅ Lightweight real-time student model
- ✅ Multi-loss training: MSE, Perceptual, Edge, and Distillation loss
- ✅ Supports high-resolution images (1920×1080)
- ✅ Real-time video frame enhancement (30–60 FPS)
- ✅ Clean training/inference pipeline

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/joel-babu10/image-sharpening-kd.git
cd image-sharpening-kd
pip install -r requirements.txt



## 🧪 Training

### 1. Train the Teacher Model (Restormer or ResUNet)

```bash
python train_teacher.py
```

### 2. Train the Student Model with Knowledge Distillation

```bash
python  train_student.py
```

> 💡 Make sure `checkpoints/teacher_model.pth` is available before training the student.

---

## 🎥 Inference on Video

```bash
python video_sharpen.py --input data/input_video.mp4 --output results/output_video.mp4
```

---

## 📁 Directory Structure

```
.
├── checkpoints/         # Saved model weights (*.pth)
├── data/                # Input/output image or video files
├── models/              # Teacher and student model definitions
├── results/             # Output sharpened images/videos
├── utils/               # Loss functions, metrics, etc.
├── main.py              # Training script
├── video_sharpen.py     # Inference script
├── requirements.txt     # Project dependencies
└── README.md            # Project overview
```

---

## 📊 Evaluation Metrics

| Model         | SSIM  | PSNR | FPS    |
|---------------|-------|------|--------|
| Teacher       | 0.9605 | 29.5 | 10–12  |
| Student (KD)  | 0.9459 | 28.3 | 35–60  |

- **SSIM (Structural Similarity):** Higher is better  
- **PSNR (Peak Signal-to-Noise Ratio):** Higher is better  
- **FPS (Frames Per Second):** Measured on 1080p input  

**MOS (Mean Opinion Score):** 4.4 / 5 (from subjective evaluation)

---

## 📈 Example Outputs

![Student Output](results//teacher_student_comparison/comparison_004.png)


> Replace the placeholders with your real outputs stored in the `results/` folder.

---

## 👥 Authors

- **Joel Babu** — Model training, architecture, experiments  
- **Hridya R Kurup1** — Loss functions, evaluation metrics  
- **Irene Anna Oommen 2** — Inference pipeline and optimization

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙋 Contributing

We welcome contributions! Feel free to fork, improve performance, or expand support to mobile/webcam use cases.
