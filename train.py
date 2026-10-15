import torch 
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

# get the dataset
df = pd.read_csv("student_data.csv")
print(df.head())

# if duplicate value exists then remove it
if df.duplicated().any():
    df = df.drop_duplicates()

# split the dataset into input_dataset and output_dataset and then split them it 80/20
x = df.drop("passed",axis=1)
y = df["passed"]

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42
)

# data pre-processing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

numeric_columns = x_train.select_dtypes(include=["number"]).columns.tolist()
categorical_column = x_train.select_dtypes(include=["object","category","bool"]).columns.tolist()

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
test_dataLoader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# create the FNN architecture by calling model.py
from model import create_model
input_features = x_train_tensor.shape[1]
model = create_model(input_features)

# set attributes for training the model
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
epochs = 100

#start training the model
model.train()
for epoch in range(epochs):
    for batch_features, batch_labels in train_dataLoader:
        # caluclate the prediction
        prediction = model(batch_features)
        #calculate the losss
        loss = criterion(prediction, batch_labels)
        # set the gradient to 0
        optimizer.zero_grad
        # Backpropogate to update the weight and bias
        loss.backward()
        # calculate the gradient when doing backpropogate
        optimizer.step()

model.eval()

with torch.no_grad():
    logits = model(x_test_tensor)
    probabilities = torch.sigmoid(logits)
    predicted_classes = (probabilities >= 0.5).int().cpu().numpy().ravel()
    
    if predicted_classes >=0.5:
        print("Pass")
    print("Fail")




# Save the model
torch.save(
    {
        "input_features": input_features,
        "model_state_dict": model.state_dict()
    },
    "my_model.pt"
)

# Store the preprocessor and encoder records
import joblib
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(encoder, "label_encoder.pkl")