import streamlit as st
import requests

st.set_page_config(page_title="ML Image Classifier", page_icon="🏠", layout="centered")

# ── Esto lo hemos añadido para poner un fondo un poco más bonito ───────────────────────────────────────────────────
st.markdown("""
    <style>
        .stApp {
            background-color: #a8d5c2;  /* verde agua */
            color: #1F4E79;             /* letra azul oscura */
        }
        .stButton>button {
            background-color: #2E75B6;
            color: white;
            border-radius: 8px;
            padding: 0.5em 2em;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #1F4E79;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────
st.title("🏠 ML Image Classifier")
st.markdown("### ¡Bienvenido a nuestra página!")
st.write("Upload an image to get a prediction from our API.")

st.divider()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

        with st.spinner('Waiting for API response...'):
            try:
                response = requests.post("http://localhost:8000/predict", files=files)
                result = response.json()

                st.divider()
                st.success(f"Prediction: {result['label']}")
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")

                st.subheader("Class Probabilities")
                probs = result["probabilities"]

                import pandas as pd
                df = pd.DataFrame({
                    "Class": [p["class"] for p in probs],
                    "Probability (%)": [round(p["probability"] * 100, 2) for p in probs]
                })
                df = df.sort_values("Probability (%)", ascending=True)
                st.bar_chart(df.set_index("Class"))

            except requests.exceptions.ConnectionError:
                st.error("Error connecting to API. Make sure fastapi_backend.py is running.")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")