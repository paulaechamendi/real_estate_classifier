from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from model import load_model, predict_image

app = FastAPI(title="Image Classifier API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargamos nuestro modelo
model, metadata = load_model(
    model_path="model_final.pth",
    metadata_path="model_metadata.json"
)

@app.get("/health")
def health():
    return {"status": "ok", "model": metadata["backbone"], "num_classes": metadata["num_classes"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # esto es para asegurar que el archivo subido es una imagen y no otro tipo de archivo, y que no esté vacío
    allowed = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=422, detail=f"Formato no soportado: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="El archivo está vacío.")

    try:
        predicted_class, confidence, probabilities = predict_image(image_bytes, model, metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {str(e)}")
    

    return {
        "filename"       : file.filename,
        "label"          : predicted_class,
        "confidence"     : round(confidence, 4),
        "probabilities"  : probabilities,
        "status"         : "success"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

