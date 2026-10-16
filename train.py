import torch 
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

# get the dataset
df = pd.read_csv("student_data.csv")

# if duplicate value exists then remove it
if df.duplicated().any():
    df = df.drop_duplicates()

# split the dataset into input_dataset and output_dataset and then split them it 80/20
x = df.drop("passed",axis=1)
y = df["passed"]

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# data pre-processing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

numeric_columns = x_train.select_dtypes(include=["number"]).columns.tolist()
categorical_column = x_train.select_dtypes(include=["object","category","bool","str"]).columns.tolist()

numeric_pipelines = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("numeric",numeric_pipelines, numeric_columns),
    ("categorical",categorical_pipeline, categorical_column)
])

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
# use label encoder for converting yes/no to number/float
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Convert the data into tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1)

# Mini-Bacth Loading method
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
train_dataLoader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataLoader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# check if the gpu is availaible or not, otherwise cpu get assigned
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device assigned : {device}")

# create the FNN architecture by calling model.py
from model import create_model
input_features = x_train_tensor.shape[1]
model = create_model(input_features)
# move the model to the gpu if availaible otherwise cpu get assigned
model = model.to(device)

# set attributes for training the model
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-4)
epochs = 100

torch.manual_seed(42)

#start training the model
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for batch_features, batch_labels in train_dataLoader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        logits = model(batch_features)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_features.size(0)

# start testing the model
model.eval()
correct1, total1 = 0, 0
# do this in testing data
with torch.no_grad():
    for batch_features, batch_labels in test_dataLoader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        probs = torch.sigmoid(model(batch_features))
        preds = (probs > 0.5).float()
        correct1 += (preds == batch_labels).sum().item()
        total1 += batch_labels.size(0)
print(f"Test accuracy in testing data: {correct1/total1:.4f}")

correct2, total2 = 0, 0
# do this in testing data
with torch.no_grad():
    for batch_features, batch_labels in train_dataLoader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        probs = torch.sigmoid(model(batch_features))
        preds = (probs > 0.5).float()
        correct2 += (preds == batch_labels).sum().item()
        total2 += batch_labels.size(0)
print(f"Test accuracy in traning data: {correct2/total2:.4f}")
    
    

# save the model and its important parameter
import joblib 
torch.save(
    {"state_dict": model.state_dict(), "input_features": input_features},
    "model.pth"
)
joblib.dump(preprocessor, "preprocessor.joblib")
joblib.dump(encoder, "label_encoder.joblib")
print("Model, preprocessor and label encoder saved.")