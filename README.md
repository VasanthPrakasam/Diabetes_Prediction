# Diabetics

This project utilizes machine learning to predict whether a person has diabetes based on various health indicators. The model is trained with the dataset (around 95000 records) and aims to identifying high-risk individuals.


Column Name	Description
Gender	            :  Gender of the individual (Male/Female/Others)
Age	                :  Age in years
Hypertension	      :  Whether the person has high blood pressure
Heart Disease	      :  Whether the person has a history of heart disease
Smoking History	    :  Smoking status (Never smoked, Current smoker, etc.)
HBA1c               :  Level	Hemoglobin A1c level Average blood sugar level over past 3 months
BMI	                :  Body Mass Index (Weight-to-height ratio)
Glucose Level	      :  Blood glucose concentration (mg/dL)
Diabetes            :  Perdicting Diabetes

🤖 Machine Learning Approach
Data Preprocessing
Handled missing values
Encoded categorical variables (e.g., Gender, Smoking History).
Normalized numerical values to ensure model stability.

Model Selection
The dataset was split into training (75%) and testing (25%) sets.
We used a Decision Tree Classifier for prediction due to its ability to handle complex decision boundaries and provide interpretability in medical applications.

3Model Training & Evaluation
Algorithm Used: Decision Tree Classifier
Performance Metrics:
✅ Accuracy: 95%
✅ Precision:0 (Non-Diabetic): 0.97 (97%) and 1 (Diabetic): 0.74 (74%)
✅ Recall: 0 (Non-Diabetic): 0.98 (98%) and 1 (Diabetic): 0.73 (73%)
✅ F1-score: 0 (Non-Diabetic): 0.97 (97%) and 1 (Diabetic): 0.73 (73%)

