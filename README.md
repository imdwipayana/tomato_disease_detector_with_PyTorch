# 🍅 Tomato Disease Detector

Detect common tomato leaf diseases using a deep learning model built with **PyTorch + MobileNetV2**, deployed with **Streamlit**.

This tool helps farmers, students, and researchers identify tomato plant health conditions from leaf images.

The complete source code is available in [this repository](https://github.com/imdwipayana/tomato_disease_detector_with_PyTorch), and the dataset can be found [here](https://www.kaggle.com/datasets/ashishmotwani/tomato).

---

## 🌿 How to Use

1. Upload a photo of a **tomato leaf** (JPG/PNG format) or capture one using your **camera**.
2. Wait a few seconds while the model analyzes the image.
3. The app will display:
   - The **predicted disease** (or “Healthy”),
   - **Confidence level**, and
   - **Prevention tips**.

---

## 🧠 Model Information

- **Architecture:** MobileNetV2 (Transfer Learning)
- **Framework:** PyTorch
- **Input Size:** 224 × 224 pixels
- **Output Classes:**  
  - Bacterial Spot  
  - Early Blight  
  - Late Blight  
  - Leaf Mold  
  - Septoria Leaf Spot  
  - Spider Mites  
  - Target Spot  
  - Tomato Yellow Leaf Curl Virus  
  - Tomato Mosaic Virus  
  - Powdery Mildew  
  - Healthy  

The model was trained on tomato leaf datasets collected from open agricultural image repositories.

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**.  
It should **not** be used as a substitute for professional agricultural advice.  
Always consult an expert before applying treatments to crops.

---

## 📸 Example Images

You can test the app with your own tomato leaf photo, or use sample images from the [`examples`](https://github.com/imdwipayana/tomato_disease_detector_with_PyTorch/tree/main/examples) folder.

---

### 👨‍💻 Author

Developed by **Eka Dwipayana**  
🔗 [Tomato Disease Detector on Streamlit Cloud](https://tomatodiseasedetectorwithpytorch.streamlit.app/)  
🔗 [LinkedIn](https://www.linkedin.com/in/eka-dwipayana/)
