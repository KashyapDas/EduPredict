import torch
import joblib
import pandas as pd
from model import create_model

# load everything once when the file is imported
checkpoint = torch.load("model.pth")
preprocessor = joblib.load("preprocessor.joblib")
label_encoder = joblib.load("label_encoder.joblib")

model = create_model(checkpoint["input_features"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()


def predict_student(study_hrs, attendance, gender):
    # column names must match the CSV used in training exactly
    data = pd.DataFrame([{
        "study_hrs": study_hrs,
        "attendance": attendance,
        "gender": gender
    }])

    x = preprocessor.transform(data)
    x_tensor = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(x_tensor)).item()

    pred = int(prob > 0.5)
    return "Pass" if pred == 1 else "Fail"