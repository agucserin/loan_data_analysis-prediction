from local_train_dl import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from matplotlib.ticker import PercentFormatter

import matplotlib.pyplot as plt
import seaborn as sns

PATH = 'G:/내 드라이브/2024 Summer/code/project_federated/'

# Load and preprocess loan data
df = pd.read_csv(PATH + 'dataset.csv')

new_df = df[:].copy()

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

new_df['loan_status'] = new_df['loan_status'].apply(transform_loan_status)

"""
# 데이터셋 요약 통계 출력
summary = new_df.describe(include='all')
# 요약 통계를 summary.csv로 저장
summary.to_csv('summary.csv', encoding='utf-8-sig')

print("Summary saved to summary.csv")"""

#------------------------------------------------------------------------------------------#

# Principal 값을 100 단위로 묶어서 범주형으로 변환
bins = range(int(new_df['Principal'].min()), int(new_df['Principal'].max()) + 200, 100)
labels = [f'{i}' for i in bins[:-1]]
new_df['Principal_binned'] = pd.cut(new_df['Principal'], bins=bins, labels=labels, right=False)

# 첫 번째 플롯: 빈도수 막대 그래프
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Principal_binned', hue='loan_status', data=new_df, palette='Set1')
plt.title('Frequency bar plot')
plt.xlabel('Principal (Binned)')
plt.ylabel('Count')
plt.xticks(rotation=45)

# 두 번째 플롯: 비율 막대 그래프
plt.subplot(1, 2, 2)
df_ratio = new_df.groupby(['Principal_binned', 'loan_status'], observed=True).size().reset_index(name='count')
df_ratio['rate'] = df_ratio['count'] / df_ratio.groupby('Principal_binned', observed=True)['count'].transform('sum')

df_ratio_pivot = df_ratio.pivot(index='Principal_binned', columns='loan_status', values='rate')
df_ratio_pivot.plot(kind='bar', stacked=True, color=['#4575b4', '#d73027'], ax=plt.gca())

plt.title('Ratio bar plot')
plt.xlabel('Principal (Binned)')
plt.ylabel('Rate')
plt.xticks(rotation=45)

# 레이아웃 설정 및 플롯 출력
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------#

# terms를 순서가 있는 범주형으로 변환 (7, 15, 30 순서로)
new_df['terms'] = new_df['terms'].astype(str)
new_df['terms'] = pd.Categorical(new_df['terms'], categories=['7', '15', '30'], ordered=True)

# 첫 번째 플롯: 빈도수 막대 그래프
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='terms', hue='loan_status', data=new_df, palette='Set1')
plt.title('Frequency bar plot by terms')
plt.xlabel('Terms')
plt.ylabel('Count')

# 두 번째 플롯: 비율 막대 그래프
plt.subplot(1, 2, 2)
sns.histplot(data=new_df, x='terms', hue='loan_status', multiple='fill', shrink=0.5, palette='Set1')
plt.ylim(0, 1)  # y축을 1.0으로 설정
plt.title('Ratio bar plot by terms')
plt.ylabel('Rate')
plt.xlabel('Terms')

# 레이아웃 설정 및 플롯 출력
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------#

# 첫 번째 플롯: 빈도수 막대 그래프
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=new_df, x='effective_date', shrink=0.8, hue='loan_status', multiple='dodge', palette='Set1')
plt.xticks(rotation=45)
plt.title('Frequency bar plot by effective date')
plt.xlabel('Effective date')
plt.ylabel('Count')
plt.legend(title='Loan Status', labels=['Fail', 'Success'])

# 두 번째 플롯: 비율 막대 그래프
plt.subplot(1, 2, 2)
sns.histplot(data=new_df, x='effective_date', hue='loan_status', multiple='fill', palette='Set1')
plt.xticks(rotation=45)
plt.title('Ratio bar plot by effective date')
plt.xlabel('Effective date')
plt.ylabel('Rate')
plt.legend(title='Loan Status', labels=['Fail', 'Success'])

# 플롯 출력
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------#

new_df['education'] = new_df['education'].apply(transform_education)
new_df['education'] = pd.Categorical(new_df['education'], 
                    categories=["High School or Below", "college", "Bechalor"], ordered=True)

# 첫 번째 플롯: 빈도수 막대 그래프
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=new_df, x='education', hue='loan_status', shrink=0.7, multiple='stack', palette='Set1')
plt.title('Frequency bar plot by education')
plt.xlabel('Education')
plt.ylabel('Count')

# 두 번째 플롯: 비율 막대 그래프
plt.subplot(1, 2, 2)
sns.histplot(data=new_df, x='education', hue='loan_status', shrink=0.7, multiple='fill', palette='Set1')
plt.title('Ratio bar plot by education')
plt.xlabel('Education')
plt.ylabel('Rate')

# 플롯 출력
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------#

# 첫 번째 플롯: 빈도수 막대 그래프
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=new_df, x='Gender', hue='loan_status', shrink=0.5, multiple='stack', palette='Set1')
plt.title('Frequency bar plot by gender')
plt.xlabel('Gender')
plt.ylabel('Count')

# 두 번째 플롯: 비율 막대 그래프
plt.subplot(1, 2, 2)
sns.histplot(data=new_df, x='Gender', hue='loan_status', shrink=0.5, multiple='fill', palette='Set1')
plt.title('Ratio bar plot by gender')
plt.xlabel('Gender')
plt.ylabel('Rate')

# 플롯 출력
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------#

new_df['effective_date'] = pd.to_datetime(new_df['effective_date'], format='%m/%d/%Y')
new_df['due_date'] = pd.to_datetime(new_df['due_date'], format='%m/%d/%Y')
new_df['period'] = new_df['due_date'].dt.dayofyear - new_df['effective_date'].dt.dayofyear

# 첫 번째 플롯: 빈도수 막대 그래프
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=new_df, x='period', hue='loan_status', multiple='dodge', palette='Set1')
plt.title('Frequency bar plot by period')
plt.xlabel('effective_date ~ due_date')
plt.ylabel('Count')

# 두 번째 플롯: 비율 막대 그래프
plt.subplot(1, 2, 2)
sns.histplot(data=new_df, x='period', hue='loan_status', multiple='fill', palette='Set1')
plt.title('Ratio bar plot by period')
plt.xlabel('effective_date ~ due_date')
plt.ylabel('Rate')

# 플롯 출력
plt.tight_layout()
plt.show()

