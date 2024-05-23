**Heart Disease Prediction using Machine Learning (Logistic Regression)**

**Overview**
This project aims to predict the presence of heart disease in a patient using machine learning techniques, specifically logistic regression. The model is trained on a dataset containing various health metrics and patient data. By analyzing these features, the model can predict whether a patient is likely to have heart disease.

**Dataset**
The dataset used in this project is the Heart Disease UCI dataset, which contains 303 records with 14 attributes. The attributes include patient information such as age, sex, chest pain type, resting blood pressure, cholesterol levels, and more.

**Attributes**
Age: Age of the patient
Sex: Sex of the patient (1 = male; 0 = female)
CP: Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)
Trestbps: Resting blood pressure (in mm Hg)
Chol: Serum cholesterol in mg/dl
Fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
Restecg: Resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy)
Thalach: Maximum heart rate achieved
Exang: Exercise-induced angina (1 = yes; 0 = no)
Oldpeak: ST depression induced by exercise relative to rest
Slope: The slope of the peak exercise ST segment (0 = upsloping; 1 = flat; 2 = downsloping)
Ca: Number of major vessels (0-3) colored by fluoroscopy
Thal: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)
Target: Diagnosis of heart disease (0 = no disease; 1 = disease)
Requirements
Python 3.x
NumPy
Pandas
Scikit-learn
Matplotlib (optional, for visualization)
Jupyter Notebook (optional, for running and testing code in notebook format)

You can install the required libraries using pip:

bash
Copy code
pip install numpy pandas scikit-learn matplotlib jupyter




**python**
Copy code
import pandas as pd

# Load the dataset
data = pd.read_csv('data/heart.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())
Step 2: Feature Engineering
Prepare the data for model training by scaling numerical features, encoding categorical features, and splitting the data into training and testing sets.

python
Copy code
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Step 3: Model Training
Train a logistic regression model using the preprocessed data.

python
Copy code
from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)
Step 4: Model Evaluation
Evaluate the trained model using metrics such as accuracy, precision, recall, and F1-score.

python
Copy code
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


**Conclusion**
This project demonstrates the process of building a logistic regression model to predict heart disease. By following the steps outlined above, you can replicate the results and further improve the model by tuning hyperparameters, trying different machine learning algorithms, or incorporating additional data preprocessing techniques.

**References**
UCI Machine Learning Repository: Heart Disease Data Set
Scikit-learn Documentation



