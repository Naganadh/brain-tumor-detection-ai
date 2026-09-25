# Brain Tumor Detection & Computer Vision Localization AI

An advanced deep learning and computer vision clinical diagnostic system for multi-class Brain MRI Tumor Detection, Localization, Cancer Staging, Nutritional Support Guidance, and Medical Report Generation.

---

## 🌟 Key Features

1. **Multi-Class Deep Learning Classification**:
   - Powered by a transfer-learning **VGG16 Convolutional Neural Network (CNN)** trained on MRI scans.
   - Accurately classifies:
     - **Glioma Tumor**
     - **Meningioma Tumor**
     - **Pituitary Neuroendocrine Tumor (PitNET)**
     - **Healthy Brain (No Tumor)**

2. **Vivid Red Lesion Localization (No Bounding Box)**:
   - Uses adaptive computer vision segmentation to isolate the focal lesion on brain parenchyma.
   - Highlights the tumor directly with a **translucent vivid red overlay (`(0, 0, 245)`)** and glowing red perimeter contour lines.
   - **No intrusive bounding boxes**, preserving underlying MRI brain tissue texture and anatomical visualization.

3. **Detailed Pathological Profiles & WHO Cancer Staging**:
   - Comprehensive pathological explanation of origin (glial cells, arachnoid cells, pituitary adenohypophysis).
   - In-depth **World Health Organization (WHO) CNS Tumor Grading Classification (Grades I to IV)**.
   - Hallmark clinical symptoms and molecular diagnostic markers (IDH1/2, 1p/19q, MGMT, NF2).

4. **Nutritional Guidance & Beneficial Anti-Tumor Foods**:
   - Evidence-based complementary nutrition (Antioxidant berries, Cruciferous vegetables, Omega-3 fatty acids, Curcumin/Turmeric with Piperine, Green Tea/Matcha EGCG, Leafy greens).
   - Guidance on foods to strictly limit or avoid (refined sugars feeding the Warburg effect, processed meats, trans-fats).

5. **Suggested Medical Therapeutics & Pharmacotherapy**:
   - Recommended medication categories (Chemotherapy such as Temozolomide, anti-VEGF biologics like Bevacizumab, anti-edema steroids like Dexamethasone, Dopamine D2 agonists like Cabergoline, anticonvulsants like Levetiracetam).
   - Detailed biological mechanisms and clinical indications.

6. **Prominent Emergency Medical Safety Notice**:
   - High-visibility safety alert:
     > ⚠️ **CRITICAL MEDICAL WARNING:** All medications require direct neuro-oncologist prescription.
     > 🚨 **IN CASE OF ANY SIDE EFFECTS, PLEASE CONTACT YOUR DOCTOR OR EMERGENCY MEDICAL SERVICES IMMEDIATELY.**

7. **Downloadable Multi-Page Clinical PDF Diagnostic Reports**:
   - Generates comprehensive PDF reports with comparative scans, probability distributions, WHO cancer staging tables, nutritional guides, medical warnings, and physician signature sections.

---

## 🚀 Running Locally

### Prerequisites
- Python 3.10+ (Python 3.11 / 3.12 / 3.13 supported)

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python main.py
```
Or double-click `run.bat` on Windows.  
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## ☁️ Live Cloud Hosting & Deployment Guide

This project is pre-configured for instant zero-configuration deployment on leading cloud platforms:

### Option A: Render.com (Recommended Free Hosting)
1. Push your repository to GitHub.
2. Sign in to [Render.com](https://render.com) and click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Render will automatically detect `render.yaml` or you can configure:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 main:app`
5. Click **Create Web Service**. Your app will be live with a free `https://<app-name>.onrender.com` URL!

### Option B: Hugging Face Spaces (Free 16GB RAM CPU Tier - Ideal for ML)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Select **Docker** as the Space SDK.
3. Push or connect this repository. The included `Dockerfile` will build and host your application automatically on port 7860!

### Option C: Railway.app
1. Go to [Railway.app](https://railway.app) and click **New Project** -> **Deploy from GitHub repo**.
2. Select this repository. Railway will detect the `Procfile` / `Dockerfile` and deploy automatically.

---

## 📁 Repository Structure

```
├── models/
│   └── model.h5             # Trained VGG16 CNN deep learning model (68 MB)
├── templates/
│   └── index.html           # Modern clinical UI with dark/light mode
├── uploads/                 # Uploaded patient MRI scans & red-highlighted outputs
├── reports/                 # Generated clinical PDF evaluation reports
├── main.py                  # Flask web backend, image processing & PDF engine
├── requirements.txt         # Production dependencies
├── Dockerfile               # Production container image configuration
├── Procfile                 # Cloud WSGI process file for Gunicorn
├── render.yaml              # Render cloud blueprint
├── run.bat                  # One-click Windows runner
└── README.md                # Project documentation
```

---

## ⚖️ Clinical Disclaimer
This system is developed for academic, scientific research, and educational demonstration purposes. It utilizes artificial intelligence algorithms for decision-support and does not substitute professional medical diagnosis, histopathological biopsy, or radiological consultation by a board-certified physician.