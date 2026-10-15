import torch
import pandas as pd
import joblib

from model import create_model

# Load saved artifacts
checkpoint = torch.load(
    "my_model.pt",
    map_location="cpu",
    weights_only=True
)

preprocessor = joblib.load("preprocessor.pkl")
encoder = joblib.load("label_encoder.pkl")

# Recreate the dynamic architecture
model = create_model(checkpoint["input_features"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def predict_result(study_hrs, attendance, gender):

    # Use the same feature names as training
    input_df = pd.DataFrame([{
        "study_hrs": study_hrs,
        "attendance": attendance,
        "gender": gender
    }])

    # Apply the exact preprocessing learned during training
    processed_input = preprocessor.transform(input_df)

    input_tensor = torch.tensor(
        processed_input,
        dtype=torch.float32
    )

    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()

    predicted_class = int(probability >= 0.5)
    predicted_label = encoder.inverse_transform(
        [predicted_class]
    )[0]

    return {
        "prediction": predicted_label,
        "probability": round(probability * 100, 2)
    }