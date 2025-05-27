Healthcare Data Analysis Project
This repository contains a Jupyter Notebook (healthcare21.ipynb) that demonstrates a basic analysis and machine learning model application on a healthcare dataset. 
The project aims to process healthcare-related information, build a predictive model, and provide an interactive interface for model inference.

1. Project Context
This project specifically focuses on applying data science and machine learning techniques to a healthcare dataset. The dataset, likely sourced from a healthcare provider or public health records, contains a variety of features that describe patients and their medical interactions. The core objective is to move beyond simple data storage to derive actionable insights. This includes:

Understanding Patient Demographics: Analyzing age, gender, and blood type distribution within the dataset.
Medical Condition Analysis: Identifying prevalent medical conditions and their distribution among patients.
Treatment and Billing Insights: Examining patterns in admission types, medications prescribed, and billing amounts.
Predictive Modeling for Test Results: The primary machine learning goal is to predict 'Test Results' (e.g., Normal, Abnormal, Inconclusive) based on other patient attributes. This could be crucial for early diagnosis, treatment planning, or resource allocation in healthcare settings.
Interactive Application Development: To make the predictive model accessible and user-friendly, an interactive web application is developed using Gradio. This allows non-technical users to input patient data and receive instant predictions, simulating a real-world application of the model.
The overall context is to showcase how data-driven approaches can be used to extract meaningful information from complex healthcare data, ultimately aiming to support better medical decision-making or administrative processes.

2. Project Code
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

    # Create a dictionary for the input features, initialized with zeros for all dummy variables
    input_data = {col: 0 for col in X.columns}

    # Populate with provided numerical inputs
    input_data['Age'] = age
    input_data['Billing Amount'] = billing_amount
    input_data['Room Number'] = room_number

    # Set dummy variables for categorical inputs
    if gender in gender_map:
        input_data[gender_map[gender]] = 1
    if blood_type in blood_type_map:
        input_data[blood_type_map[blood_type]] = 1
    if medical_condition in medical_condition_map:
        input_data[medical_condition_map[medical_condition]] = 1
    if admission_type in admission_type_map:
        input_data[admission_type_map[admission_type]] = 1
    if medication in medication_map:
        input_data[medication_map[medication]] = 1

    # Convert the dictionary to a DataFrame in the correct order of columns
    input_df = pd.DataFrame([input_data], columns=X.columns)

    # Make prediction
    prediction = model.predict(input_df)

    # Map numerical prediction back to original labels
    if prediction[0] == 1:
        return "Normal"
    elif prediction[0] == 0:
        return "Abnormal"
    else:
        return "Inconclusive"

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

3. Key Technologies
The project utilizes the following key Python libraries:

Pandas: For data manipulation and analysis.
Scikit-learn (sklearn): For machine learning tasks, including model selection (train_test_split), logistic regression (LogisticRegression), and evaluation metrics (classification_report, accuracy_score).
Gradio: For building interactive web applications for machine learning models, enabling easy demonstration and use of the trained model.
4. Description
This project demonstrates a complete workflow for building a predictive model on a healthcare dataset. It starts with loading and inspecting the raw data, followed by essential preprocessing steps such as handling categorical variables through one-hot encoding and dropping irrelevant columns.
A logistic regression model is then trained to predict 'Test Results' based on various patient attributes. Finally, an interactive Gradio interface is built, allowing users to input patient details and receive predictions from the deployed model. 
This setup provides a practical example of how machine learning can be applied to healthcare data for predictive analytics and showcases an accessible way to interact with such models.

5. Output
The notebook produces several key outputs:

Initial Data Overview: Displays the first few rows and summary of the health dataframe, showing the raw data structure with 15 columns and 55,500 entries.
Processed Data Overview: Shows the dataframe after dropping the 'Name' column, indicating the reduced feature set.
Model Evaluation Report:
Classification Report: Provides precision, recall, f1-score, and support for each class of 'Test Results' (Normal, Abnormal, Inconclusive).
Accuracy Score: Shows the overall accuracy of the Logistic Regression model on the test set.
Gradio Interface URL: Upon launching the Gradio interface, a public URL (e.g., https://93b611607081771aea.gradio.live) is provided, allowing external access to the interactive prediction tool. This link typically expires after a week.
6. Further Research
To enhance this project, consider the following areas for further research and development:

Advanced Data Preprocessing:
Handling missing values in a more sophisticated manner (e.g., imputation techniques).
Outlier detection and treatment for numerical features.
Feature scaling (e.g., StandardScaler, MinMaxScaler) for numerical features, which can significantly improve the performance of distance-based algorithms like Logistic Regression.
Exploratory Data Analysis (EDA): Conduct a more in-depth EDA to uncover patterns, correlations, and insights within the healthcare data. This could involve visualizations, statistical tests, and subgroup analysis.
Model Selection and Hyperparameter Tuning:
Experiment with other machine learning algorithms suitable for multi-class classification (e.g., Decision Trees, Random Forests, Gradient Boosting, Support Vector Machines, Neural Networks) and compare their performance.
Perform hyperparameter tuning using techniques like Grid Search or Random Search to optimize the chosen model's performance.
Feature Importance Analysis: Investigate which features contribute most to the model's predictions. This can provide valuable insights into the healthcare factors influencing test results.
Interpretability and Explainability: Explore techniques to make the model's predictions more interpretable, especially crucial in healthcare contexts (e.g., SHAP, LIME).
Unbalanced Datasets: If the 'Test Results' classes are imbalanced, apply techniques like SMOTE, oversampling, or undersampling to improve model performance for minority classes.
Deployment and MLOps: Explore more robust deployment strategies beyond a temporary Gradio share link, such as deploying to cloud platforms (AWS, GCP, Azure) or using containerization (Docker) for consistent environments. Implement MLOps practices for continuous integration, deployment, and monitoring.
Real-time Data Integration: Investigate how real-time healthcare data could be integrated for continuous model updates and predictions.
Ethical Considerations: Address potential biases in the data and model, and discuss ethical implications of deploying such a model in a healthcare setting.
