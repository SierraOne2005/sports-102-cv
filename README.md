
# Sports-102 Image Classification  

## 📌 Overview  
This project tackles **multi-class sports image classification** using the **Sports-102 dataset**, which contains images across **102 different sports categories**.  

We leverage **state-of-the-art deep learning architectures** — **ConvNeXt Large** and **SwinV2 Large** — to build highly accurate classifiers. Due to Kaggle’s 12-hour session constraints, the training pipeline is designed to **resume seamlessly across multiple sessions**. Finally, we use **Test-Time Augmentation (TTA)** and an **ensemble of both models** to achieve a strong final accuracy.  

---

## 🚀 Features  
- **Large-scale sports classification** (102 categories).  
- **Transfer learning** with ImageNet-22K pre-trained weights.  
- **Resumable training** across Kaggle sessions (checkpointing).  
- **Mixed precision training** with `torch.cuda.amp` for speed & efficiency.  
- **Data augmentations** including Mixup and standard torchvision transforms.  
- **EMA (Exponential Moving Average)** model updates for stability.  
- **TTA + Ensemble** for final evaluation.  

---

## 📂 Dataset  
- **Name:** Sports-102 (V2)  
- **Structure:**  
  ```
  Sports102_V2/
  ├── train/
  │   ├── class_1/
  │   ├── class_2/
  │   └── ...
  └── test/
      ├── class_1/
      ├── class_2/
      └── ...
  ```  
- **Loading:**  
  ```python
  from torchvision import datasets, transforms
  train_ds = datasets.ImageFolder(os.path.join(DATA_PATH, "train"), transform=train_tf)
  val_ds   = datasets.ImageFolder(os.path.join(DATA_PATH, "test"), transform=val_tf)
  ```

---

## 🛠️ Tech Stack  
- **Framework:** PyTorch  
- **Models:** ConvNeXt Large, SwinV2 Large  
- **Libraries:**  
  - `torch`, `torchvision`, `timm`  
  - `numpy`, `tqdm`  
  - `torch.cuda.amp` (mixed precision)  
  - `ModelEmaV2` (EMA model updates)  

---

## 📑 Training Strategy  

### Week 1: ConvNeXt Large  
1. Configure backbone → `convnext_large.fb_in22k_ft_in1k`  
2. Train across 3 sessions (~8–10 hrs each).  
3. Save and resume checkpoints between sessions.  

### Week 2: SwinV2 Large  
1. Configure backbone → `swinv2_large_window12_384.ms_in22k_ft_in1k`  
2. Repeat multi-session training.  
3. Attach final ConvNeXt model for evaluation.  

### Final Step: Ensemble Evaluation  
- Load both models.  
- Apply **Test-Time Augmentation (TTA)**.  
- Perform **Ensemble Inference**.  
- Report final accuracy.  

---

## ▶️ Running the Notebook  
1. Clone this repo:  
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Open in **Kaggle/Colab/Jupyter**.  
3. Set dataset path in notebook:  
   ```python
   DATA_PATH = "/kaggle/input/sports-102/Sports102_V2"
   ```
4. Follow the step-by-step notebook cells:  
   - **Cell 1:** Install libraries  
   - **Cell 2:** Imports & config  
   - **Cell 3–5:** Resumable training  
   - **Cell 6:** Final ensemble evaluation  

---

## 📊 Results  
- **ConvNeXt Large:** High single-model accuracy (~XX%, replace with your logs).  
- **SwinV2 Large:** Comparable single-model accuracy (~XX%).  
- **Ensemble + TTA:** Final accuracy = **XX%** (best performance).  

*(Fill in actual accuracy after evaluation.)*  

---

## 📌 Future Work  
- Explore lighter models for faster training.  
- Deploy trained ensemble as an API/web app.  
- Experiment with dataset balancing and additional augmentations.  

---

## 🙏 Acknowledgements  
- **Dataset:** [Sports-102 Dataset](https://www.kaggle.com/)  
- **Models:** Pretrained weights from [timm](https://github.com/huggingface/pytorch-image-models).  
- **Training:** Kaggle GPU sessions.  

---

## 📜 License  
This project is released under the **MIT License**.  
