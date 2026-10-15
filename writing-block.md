# 🎓 Student Pass Prediction Using PyTorch

A machine learning project that predicts whether a student will pass or fail based on study hours, attendance, and gender using a **Feedforward Neural Network (FNN)** built with PyTorch.

## 📌 Project Overview

The goal of this project is to develop a binary classification model that learns patterns from student data and predicts the likelihood of passing or failing.

The project uses PyTorch to build and train the neural network and Scikit-learn to preprocess the dataset before passing it to the model.

## ✨ Features

- **Feedforward Neural Network:** Implements a neural network using PyTorch.
- **Dynamic Model Architecture:** Automatically determines the number of input features from the preprocessed training data.
- **Data Preprocessing:** Handles missing values, scales numerical features, and encodes categorical features.
- **Categorical Encoding:** Uses One-Hot Encoding for categorical input features and Label Encoding for the target.
- **Model Training:** Trains the network using backpropagation and the Adam optimizer.
- **Model Evaluation:** Evaluates performance on a separate test dataset.
- **Model Persistence:** Saves the trained model, preprocessing pipeline, and label encoder.
- **Prediction Pipeline:** Accepts student details and returns a predicted pass/fail result.

## 🛠️ Tech Stack

- **Python**
- **PyTorch** — Neural network architecture and training
- **Pandas** — Dataset loading and manipulation
- **Scikit-learn** — Data preprocessing, train-test splitting, and evaluation
- **Joblib** — Saving and loading preprocessing objects

## 📂 Project Structure

```text
student_prediction/
│
├── student_data.csv       # Dataset
├── model.py               # Neural network architecture
├── train.py               # Data preprocessing and model training
├── functionCall.py        # Prediction function
├── index.py               # Entry point for predictions
│
├── my_model.pt            # Saved model checkpoint
├── preprocessor.pkl       # Saved preprocessing pipeline
├── label_encoder.pkl      # Saved target label encoder
│
└── README.md              # Project documentation
```

The three saved artifacts are generated when you run the training script. They do not need to be committed to GitHub if you choose to regenerate them during setup.

## 📊 Dataset Description

The dataset is stored in `student_data.csv`.

| Feature | Type | Description |
|---|---|---|
| `study_hrs` | Numerical | Number of hours spent studying |
| `attendance` | Numerical | Student attendance percentage |
| `gender` | Categorical | Student gender |
| `passed` | Target | Whether the student passed (`yes` or `no`) |

### Sample Data

| study_hrs | attendance | gender | passed |
|---:|---:|---|---|
| 10.0 | 95 | female | yes |
| 5.4 | 64 | female | no |
| 8.8 | 40 | male | yes |
| 2.7 | 58 | female | yes |
| 5.7 | 70 | male | yes |

The dataset is used to train a binary classification model.

## 🧠 Model Architecture

The project uses a Feedforward Neural Network (FNN) with fully connected layers.

```text
Input Layer
    ↓
Linear(input_features, 16)
    ↓
ReLU
    ↓
Linear(16, 8)
    ↓
ReLU
    ↓
Linear(8, 1)
    ↓
Output Logit
    ↓
Sigmoid
    ↓
Pass Probability
```

### Architecture Explanation

- **Input Layer:** Accepts the features produced by the preprocessing pipeline.
- **First Hidden Layer:** Contains 16 neurons and uses the ReLU activation function.
- **Second Hidden Layer:** Contains 8 neurons and uses ReLU.
- **Output Layer:** Produces one logit for binary classification.
- **Sigmoid Function:** Converts the output logit into a probability between 0 and 1 during prediction.

The model uses `BCEWithLogitsLoss` as its loss function and the Adam optimizer to update the network weights.

## ⚙️ Data Preprocessing

The project uses a Scikit-learn preprocessing pipeline.

1. **Duplicate Removal:** Removes duplicate rows from the dataset.
2. **Train-Test Split:** Divides the dataset into training and testing subsets using an 80:20 ratio.
3. **Missing Value Handling:** Fills missing numerical values using the median and categorical values using the most frequent value.
4. **Numerical Scaling:** Applies `StandardScaler` to numerical features.
5. **Categorical Encoding:** Applies `OneHotEncoder` to categorical input features.
6. **Target Encoding:** Converts the pass/fail labels into numerical values.
7. **Tensor Conversion:** Converts the processed features and labels into PyTorch tensors.

The preprocessing pipeline is fitted on the training data and then reused to transform the test data and future prediction inputs.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd student_prediction
```

Replace `<your-repository-url>` with your actual GitHub repository URL.

### 2. Install Dependencies

```bash
pip install torch pandas scikit-learn joblib
```

### 3. Train the Model

Run the training script:

```bash
python train.py
```

The script preprocesses the dataset, trains the neural network, evaluates its test accuracy, and saves the model and preprocessing artifacts.

Expected output includes training loss for each epoch and the final test accuracy.

### 4. Make Predictions

After training, run:

```bash
python index.py
```

The prediction script loads the saved artifacts and predicts the outcome using the supplied student details.

## 💻 Example Prediction

The following example demonstrates the prediction function:

```python
from functionCall import predict_result

result = predict_result(
    study_hrs=8,
    attendance=85,
    gender="male"
)

print(result)
```

Example output format:

```python
{
    "prediction": "yes",
    "probability": 78.42
}
```

*Note: The probability shown above is illustrative, not an actual result from the trained model.*

## 📈 Model Evaluation

The model evaluates its performance on the test subset using classification accuracy.

Additional metrics that can be used to assess the model include:

- **Precision:** Measures how many predicted positive cases are actually positive.
- **Recall:** Measures how many actual positive cases the model correctly identifies.
- **F1-Score:** Combines precision and recall into a single metric.
- **Confusion Matrix:** Shows correct and incorrect predictions for each class.

Because the dataset is relatively small, further testing on more diverse, unseen student data is necessary before drawing conclusions about real-world performance.

## 🔮 Future Improvements

- Increase the size and diversity of the dataset.
- Apply cross-validation for more reliable evaluation.
- Compare the neural network against Logistic Regression, Random Forest, and other baseline models.
- Add precision, recall, F1-score, and confusion matrix reporting.
- Build a REST API using FastAPI or Flask.
- Develop a frontend using React to accept student details and display predictions.
- Experiment with hyperparameter tuning to improve model performance.

## ⚠️ Limitations

- The model is trained on a small dataset and may not generalize well to unseen students.
- Predictions depend on the quality and representativeness of the training data.
- The predicted probability should not be interpreted as a guaranteed chance of passing.
- Gender-based predictions may introduce or reproduce bias and should be evaluated carefully before any real-world use.

## 👨‍💻 Author

**Kashyap Jyoti Das**

GitHub: [KashyapDas](https://github.com/KashyapDas)

---

⭐ If you find this project useful, consider giving the repository a star!