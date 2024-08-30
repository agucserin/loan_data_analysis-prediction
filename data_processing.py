from local_train_dl import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

PATH = 'G:/내 드라이브/2024 Summer/code/project_federated/'

# Load and preprocess loan data
data = pd.read_csv(PATH + 'dataset.csv')

columns_to_extract = ['loan_status', 'Principal', 'terms', 'effective_date', 'due_date', 'age', 'education', 'Gender']
extracted_data = data[columns_to_extract].copy()

extracted_data['terms'] = extracted_data['terms'].astype('category')

# 날짜 형식 변환
extracted_data['effective_date'] = pd.to_datetime(extracted_data['effective_date'], format='%m/%d/%Y')
# 'effective_day' 파생 변수 생성
extracted_data['effective_day'] = extracted_data['effective_date'].dt.strftime('%m%d.%a')
# 'effective_day'를 범주형 변수로 변환
extracted_data['effective_day'] = extracted_data['effective_day'].astype('category')


extracted_data['due_date'] = pd.to_datetime(extracted_data['due_date'], format='%m/%d/%Y')
extracted_data['due_month'] = extracted_data['due_date'].dt.month
extracted_data['due_day'] = extracted_data['due_date'].dt.dayofweek
extracted_data['due_month'] = extracted_data['due_month'].astype('category')
extracted_data['due_day'] = extracted_data['due_day'].astype('category')



extracted_data['period'] = extracted_data['due_date'].dt.dayofyear - extracted_data['effective_date'].dt.dayofyear

# loan_status 변환
def transform_loan_status(status):
    if status == 'PAIDOFF':
        return 'success'
    else:
        return 'fail'
    
def transform_education(status):
    if status == 'Bechalor' or status == 'Master or Above':
        return 'Bechalor'
    else:
        return status

extracted_data['loan_status'] = extracted_data['loan_status'].apply(transform_loan_status)
extracted_data['education'] = extracted_data['education'].apply(transform_education)

# education과 Gender를 범주형 변수로 변환
extracted_data.loc[:, 'education'] = extracted_data['education'].astype('category')
extracted_data.loc[:, 'Gender'] = extracted_data['Gender'].astype('category')

# Encode categorical variables
label_encoder = LabelEncoder()
#extracted_data['terms'] = label_encoder.fit_transform(extracted_data['terms'])
extracted_data['effective_day'] = label_encoder.fit_transform(extracted_data['effective_day'])
extracted_data['loan_status'] = label_encoder.fit_transform(extracted_data['loan_status'])
extracted_data['education'] = label_encoder.fit_transform(extracted_data['education'])
extracted_data['Gender'] = label_encoder.fit_transform(extracted_data['Gender'])

# Drop original date columns
extracted_data = extracted_data.drop(['effective_date', 'due_date'], axis=1)

# Split features and target
X = extracted_data.drop('loan_status', axis=1)
y = extracted_data['loan_status']


numeric_features = ['Principal', 'age', 'period']
categorical_features = ['terms', 'effective_day', 'due_month', 'due_day', 'education', 'Gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Transform the features using the preprocessor
X_processed = preprocessor.fit_transform(X)

# Label Encoding for the target variable
label_encoder = LabelEncoder()
y_processed = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.3, random_state=42)

# Combine X_test and y_test back into a DataFrame
test_set = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out())
test_set['loan_status'] = y_test

# Save test set
test_set.to_csv(PATH + '/test_set.csv', index=False)

# 2. Randomly split the train set into two groups
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Combine X_train_1 and y_train_1 back into a DataFrame
train_set_1 = pd.DataFrame(X_train_1, columns=preprocessor.get_feature_names_out())
train_set_1['loan_status'] = y_train_1

# Combine X_train_2 and y_train_2 back into a DataFrame
train_set_2 = pd.DataFrame(X_train_2, columns=preprocessor.get_feature_names_out())
train_set_2['loan_status'] = y_train_2

# Save train sets
train_set_1.to_csv(PATH + '/train_set_1.csv', index=False)
train_set_2.to_csv(PATH + '/train_set_2.csv', index=False)
