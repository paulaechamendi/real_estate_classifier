# Real Estate Image Classifier

Automatic classification of real estate images into 15 scene categories using transfer learning (ConvNeXt-Small) with a FastAPI backend and Streamlit frontend.

## Project Structure

```
real-estate-classifier/
├── real_estate_classifier_v2.ipynb   # Training notebook (Google Colab)
|   
├── api/
│   ├── fastapi_backend.py            # FastAPI REST API
│   ├── model.py                      # Model loading and inference
│   ├── app.py                        # Streamlit frontend
│   └── requirements.txt              # Python dependencies
└── graficas/                         # W&B training charts
```

## 🧠 Model

- **Backbone:** ConvNeXt-Small (pretrained on ImageNet-1K)
- **Classes:** Bedroom, Coast, Forest, Highway, Industrial, Inside city, Kitchen, Living room, Mountain, Office, Open country, Store, Street, Suburb, Tall building
- **Test accuracy:** 97.47% | **Macro F1:** 97.45%

>  The model weights (`model_final.pth`) are not included in this repository due to file size. Download them from Google Drive  and place them inside the `api/` folder:
> - `api/model_final.pth`   -> https://drive.google.com/file/d/1rMMUAptI_ld_TLeavO6KJPn97DOjxhb0/view?usp=drive_link
> - `api/model_metadata.json`  -> https://drive.google.com/file/d/1PosnEr26M7MiHaofBymYcG_saJQC9v7A/view?usp=drive_link 

##  Installation

```bash
cd api
pip install -r requirements.txt
```

## 🚀 Running the application

Open **two terminals** inside the `api/` folder:

**Terminal 1 — Start the API:**
```bash
python fastapi_backend.py
```
API available at: `http://localhost:8000`
Swagger docs at: `http://localhost:8000/docs`

**Terminal 2 — Start Streamlit:**
```bash
streamlit run app.py
```
App available at: `http://localhost:8501`

## 🔗 Links

- **W&B Project:** https://wandb.ai/202109933-universidad-pontificia-comillas/real-estate-classifier
- **W&B Sweep:** https://wandb.ai/202109933-universidad-pontificia-comillas/real-estate-classifier/sweeps/8ntzldx4
