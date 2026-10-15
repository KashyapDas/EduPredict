from model import load_model
import torch

model = load_model()

def predict_result(study_hrs, attendance):
    student_data = [[study_hrs, attendance]]
    student_tensor = torch.tensor(student_data, dtype=torch.float32)
    with torch.no_grad():
        logits = model(student_tensor)
        probability = torch.sigmoid(logits).item()
        if probability >= 0.5 :
            return "Pass"
        return "Fail"