"""
Carga del modelo ConvNeXt-Small y lógica de inferencia.
"""

import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io


def build_model(backbone_name: str, num_classes: int, dropout: float) -> nn.Module:
    """Reconstruye exactamente la misma clase CNN del entrenamiento."""
    if backbone_name == "convnext_small":
        base = models.convnext_small(weights=None)
        in_features = base.classifier[2].in_features
        base.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    elif backbone_name == "efficientnet_b3":
        base = models.efficientnet_b3(weights=None)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    elif backbone_name == "resnet50":
        base = models.resnet50(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Backbone desconocido: {backbone_name}")

    class CNNWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        def forward(self, x):
            return self.base_model(x)

    return CNNWrapper(base)



def load_model(model_path: str, metadata_path: str):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        backbone_name=metadata["backbone"],
        num_classes=metadata["num_classes"],
        dropout=metadata["dropout"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"[API] Modelo cargado: {metadata['backbone']} | "
          f"{metadata['num_classes']} clases | device={device}")

    metadata["device"] = device
    return model, metadata

def predict_image(image_bytes: bytes, model: nn.Module, metadata: dict):
    """Realiza la inferencia sobre una imagen en bytes.

    Args:
        image_bytes : contenido binario de la imagen
        model       : modelo PyTorch en modo eval
        metadata    : dict con class_names, img_size y normalize params

    Returns:
        predicted_class : nombre de la clase predicha
        confidence      : probabilidad de la clase predicha
        probabilities   : lista de {class, probability} ordenada de mayor a menor
    """
    img_size    = metadata["img_size"]
    mean        = metadata["normalize"]["mean"]
    std         = metadata["normalize"]["std"]
    class_names = metadata["class_names"]
    device      = metadata["device"]

    # Preprocesado idéntico al val_test_transforms del entrenamiento
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Abrir imagen
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]

    # Inferencia
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()  # [num_classes]

    # Resultado
    probs_list = probs.cpu().tolist()
    predicted_idx   = int(probs.argmax())
    predicted_class = class_names[predicted_idx]
    confidence      = probs_list[predicted_idx]

    probabilities = sorted(
        [{"class": class_names[i], "probability": round(p, 4)}
         for i, p in enumerate(probs_list)],
        key=lambda x: x["probability"],
        reverse=True
    )

    return predicted_class, confidence, probabilities
