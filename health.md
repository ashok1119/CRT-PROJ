import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import gradio as gr

# Load the dataset
health = pd.read_csv("/content/health.csv")

# Data Preprocessing
# Drop irrelevant columns
health.drop(columns=['Name'], inplace=True)

# One-hot encode categorical features
categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Medication', 'Test Results']
for col in categorical_cols:
    health = pd.get_dummies(health, columns=[col], prefix=col)

# Map 'Test Results' to numerical values and handle missing values
health['Test Results'] = health['Test Results'].map({'Normal': 1, 'Abnormal': 0, 'Inconclusive': 2})
health.drop(columns=['Date of Admission', 'Doctor', 'Hospital', 'Insurance Provider', 'Discharge Date'], inplace=True)
health = health.dropna(subset=['Test Results'])

# Define features (X) and target (y)
X = health.drop('Test Results', axis=1)
y = health['Test Results']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model (optional, for verification)
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Gradio Interface Function
def predict_test_result(age, gender, blood_type, medical_condition, billing_amount, room_number, admission_type, medication):
    # Mapping for categorical inputs to match the one-hot encoded format used in training
    gender_map = {'Male': 'Gender_Male', 'Female': 'Gender_Female', 'Other': 'Gender_Other'}
    blood_type_map = {'A+': 'Blood Type_A+', 'A-': 'Blood Type_A-', 'B+': 'Blood Type_B+', 'B-': 'Blood Type_B-',
                      'AB+': 'Blood Type_AB+', 'AB-': 'Blood Type_AB-', 'O+': 'Blood Type_O+', 'O-': 'Blood Type_O-'}
    medical_condition_map = {'Obesity': 'Medical Condition_Obesity', 'Arthritis': 'Medical Condition_Arthritis',
                             'Hypertension': 'Medical Condition_Hypertension', 'Diabetes': 'Medical Condition_Diabetes',
                             'Asthma': 'Medical Condition_Asthma', 'Cancer': 'Medical Condition_Cancer'}
    admission_type_map = {'Urgent': 'Admission Type_Urgent', 'Emergency': 'Admission Type_Emergency', 'Elective': 'Admission Type_Elective'}
    medication_map = {'Paracetamol': 'Medication_Paracetamol', 'Ibuprofen': 'Medication_Ibuprofen',
                      'Aspirin': 'Medication_Aspirin', 'Penicillin': 'Medication_Penicillin',
                      'Lipitor': 'Medication_Lipitor'}

   

# Define Gradio input components
input_components = [
    gr.Slider(minimum=0, maximum=100, step=1, label="Age"),
    gr.Dropdown(["Male", "Female", "Other"], label="Gender"),
    gr.Dropdown(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], label="Blood Type"),
    gr.Dropdown(["Obesity", "Arthritis", "Hypertension", "Diabetes", "Asthma", "Cancer"], label="Medical Condition"),
    gr.Number(label="Billing Amount"),
    gr.Number(label="Room Number"),
    gr.Dropdown(["Urgent", "Emergency", "Elective"], label="Admission Type"),
    gr.Dropdown(["Paracetamol", "Ibuprofen", "Aspirin", "Penicillin", "Lipitor"], label="Medication")
]

# Create Gradio interface
iface = gr.Interface(fn=predict_test_result, inputs=input_components, outputs="text")

# Launch the interface
iface.launch(share=True)
