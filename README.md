# 3. 데이터 분석과 연방 학습을 활용한 대출고객 분석 및 예측

## 1. 개요

### **1. 프로젝트 배경**

 최근 데이터 프라이버시와 보안에 대한 관심이 높아지면서, 민감한 데이터를 중앙 서버로 전송하지 않고도 모델을 학습할 수 있는 방법이 필요해졌습니다. 특히 금융 산업에서는 고객의 신용 정보를 보호하면서도 정확한 신용 예측을 수행하는 것이 중요합니다. 연방 학습(Federated Learning)은 이러한 문제를 해결할 수 있는 혁신적인 기술로 주목받고 있습니다.

### **2. 프로젝트 목적**

 이 프로젝트의 목적은 연방 학습을 활용하여 중앙 서버에 데이터를 공유하지 않고도 분산된 환경에서 신용 대출 예측 모델을 개발하는 것입니다. 이를 통해 데이터 유출의 위험을 최소화하면서도 정확한 예측 모델을 구축하고, 이를 실제 비즈니스 환경에서 적용 가능하게 하는 것을 목표로 합니다.

### **3. 접근 방식**

- **데이터 분석 및 전처리**: 분산된 여러 데이터셋에서 신용 대출 관련 데이터를 수집하고, 탐색적 데이터 분석(EDA)을 통해 주요 특성들을 파악한 후, 데이터를 전처리하여 모델 학습에 적합하게 변환합니다.
- **연방 학습 설계**: 각 데이터 소스(피어)가 개별적으로 모델을 학습한 후, 중앙 서버에서 모델의 가중치를 통합하여 최종 예측 모델을 구축합니다. 이 과정에서 데이터 프라이버시를 유지하면서도 학습 효율을 극대화할 수 있는 연방 학습 전략을 설계합니다.
- **모델 학습 및 평가**: 여러 딥러닝 모델을 비교하고 최적의 모델을 선택하여 연방 학습을 적용합니다. 또한 교차 검증, 하이퍼파라미터 튜닝 등을 통해 모델의 성능을 최적화합니다. 최종 모델의 성능은 정확도(Accuracy), 손실(Loss), AUC 등 다양한 평가 지표를 통해 평가됩니다.

### **4. 사용 기술 및 도구**

- **프로그래밍 언어**: Python
- **딥러닝 프레임워크**: PyTorch
- **데이터 분석 도구**: Pandas, NumPy, Scikit-learn
- **시각화 도구**: Matplotlib, Seaborn
- **분산 학습 환경**: 연방 학습 구현을 위한 PyTorch 기반 커스텀 알고리즘

### **5. 기대 효과 및 비즈니스 임팩트**

 이 프로젝트는 민감한 금융 데이터를 중앙 서버에 공유하지 않고도, 분산된 데이터 환경에서 고성능의 신용 대출 예측 모델을 구축할 수 있음을 입증합니다. 이는 금융 기관이 데이터 프라이버시를 유지하면서도 예측 모델의 정확도를 향상시킬 수 있는 방법을 제공합니다. 결과적으로, 이 프로젝트는 고객 신뢰를 유지하며 비즈니스의 경쟁력을 강화하는 데 기여할 수 있습니다.

## 2. 실험 과정

### 1. 데이터 분석 및 전처리

- **1. 원 데이터 셋 탐색**
    
     Kaggle의 open 데이터 중에서 고객의 대출 정보를 담고 있는 ‘**Loan Data’** 라는 데이터셋을 다운 받았습니다. 이 데이터셋은 각각의 고객에 대해 총 11가지 변수를 담고 있습니다. 데이터를 구성하는 11가지 변수들은 다음과 같습니다.
    
    | 변수명 | 설명 |
    | --- | --- |
    | **Loan_ID** | 각 대출에 고유하게 할당된 대출 번호입니다. 각 고객의 대출을 식별하기 위해 사용됩니다. |
    | **loan_status** | 대출의 상태를 나타냅니다. 여기에는 대출이 상환되었는지, 연체 중인지, 상환 예정인 신규 고객인지, 아니면 연체 후 상환되었는지가 포함됩니다. |
    | **Principal** | 대출 시작 시의 기본 대출 금액을 의미합니다. 이는 대출의 원금으로, 대출자가 대출 당시 받은 금액을 나타냅니다. |
    | **terms** | 대출 상환 일정의 빈도를 나타냅니다. 상환은 주간(7일), 격주 또는 월간으로 이루어질 수 있습니다. |
    | **effective_date** | 대출이 발생하고 효력이 발생한 날짜를 나타냅니다. 이는 대출 계약이 공식적으로 시작된 날을 의미합니다. |
    | **due_date** | 상환 기한을 나타냅니다. 이는 일회성 상환 일정으로, 각 대출에 하나의 상환 기한이 있습니다. |
    | **paid_off_time** | 고객이 대출을 실제로 상환한 시간을 나타냅니다. 이는 대출 상환이 완료된 시간을 기록합니다. |
    | **past_due_days** | 대출이 연체된 일수를 나타냅니다. 고객이 상환 기한을 지나 얼마나 연체되었는지를 표시합니다. |
    | **age** | 고객의 나이를 나타냅니다. |
    | **education** | 고객의 학력 수준을 나타냅니다. 고객이 어떤 교육을 받았는지를 의미합니다. |
    | **Gender** | 고객의 성별을 나타냅니다. |
    
     이번 프로젝트에서는 나머지 10개의 변수들을 입력변수로 활용하여 loan_status를 예측하는, 즉 loans_status를 target 변수로 활용하였습니다. 모델의 단순화 및 정확도 향상 위해, 원래는 3개의 class(PAIDOFF, COLLECTION, COLLECTION_PAIDOFF)였지만 기한 내에 상환이 가능한지 여부를 판단하는 binary classification으로 변형하였습니다.
    
    |  | Loan_ID | loan_status | effective_date | due_date | paid_off_time | education | Gender |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | count | 500 | 500 | 500 | 500 | 400 | 500 | 500 |
    | unique | 500 | 2 | 7 | 25 | 320 | 4 | 2 |
    | top | xqd20166231 | success | 9/11/2016 | 10/10/2016 | 9/26/2016 9:00 | college | male |
    | freq | 1 | 300 | 231 | 123 | 9 | 220 | 423 |
    
    |  | past_due_days | age | Principal | terms |
    | --- | --- | --- | --- | --- |
    | count | 200 | 500 | 500 | 500 |
    | mean | 36.01 | 31.116 | 943.2 | 22.824 |
    | std | 29.3808795 | 6.084783742 | 115.24027 | 8.0000641 |
    | min | 1 | 18 | 300 | 7 |
    | 25% | 3 | 27 | 1000 | 15 |
    | 50% | 37 | 30 | 1000 | 30 |
    | 75% | 60 | 35 | 1000 | 30 |
    | max | 76 | 51 | 1000 | 30 |
    
    ---
    
    summary 결과, 다음과 같은 결론을 내릴 수 있었습니다.
    
    - **Loan_ID** : 각 고객(데이터)마다 고유한 값을 가지며, target 변수(loan_status)와 상관관계가 없으므로 학습에서 제외하였습니다.
    - **age, Princiap, terms** 변수를 제외한 나머지 변수들은 전부 범주형 변수입니다. 그러나 양적 변수인 `terms`의 경우 종류가 총 3가지(7,15,30)밖에 없으므로 범주형 변수로 변환하는 것이 효율적이라고 판단했습니다.
    - **paid_off_time, past_due_days**: 이 변수들은 각 데이터 내에서 target 변수에 따라 존재 여부가 달라지므로, 이번 프로젝트에서는 적합하지 않은 변수로 판단하여 제외하였습니다.
    
- **2. 데이터 분석**
    
     결론에 따라 데이터 분석을 시작했습니다. 대출 데이터셋을 분석하여 주요 변수들이 대출 성공 여부에 미치는 영향을 평가하고자 하였습니다. 이를 위해 다양한 변수에 대한 탐색적 데이터 분석(EDA)을 수행하였으며, 각 변수와 대출 성공 여부 간의 관계를 시각화하였습니다.
    
    ---
    
    **<Principal 변수 분석>**
    
    ```python
    # Principal 값을 100 단위로 묶어서 범주형으로 변환
    bins = range(int(new_df['Principal'].min()), int(new_df['Principal'].max()) + 200, 100)
    new_df['Principal_binned'] = pd.cut(new_df['Principal'], bins=bins, labels=labels, right=False)
    
    # 첫 번째 플롯: 빈도수 막대 그래프
    sns.countplot(x='Principal_binned', hue='loan_status', data=new_df, palette='Set1')
    
    # 두 번째 플롯: 비율 막대 그래프
    df_ratio = new_df.groupby(['Principal_binned', 'loan_status'], observed=True).size().reset_index(name='count')
    df_ratio['rate'] = df_ratio['count'] / df_ratio.groupby('Principal_binned', observed=True)['count'].transform('sum')
    ```
    
     **Principal** 변수는 대출의 기본 원금을 나타냅니다. 이 변수를 100 단위로 구간화하여 대출 금액의 크기에 따른 대출 상환 상태를 분석했습니다.
    
    - **빈도수 분석**: Principal 변수를 100 단위로 구간화하여 성공 및 실패 빈도수를 시각화하였습니다.
    - **비율 분석**: 동일한 데이터를 사용하여 각 구간에서 대출 성공과 실패의 비율을 시각화하였습니다.
    
     이 분석을 통해, 대출 원금의 크기와 상환 성공 여부 간의 관계를 직관적으로 파악할 수 있었습니다.
    
    ![Figure_1](https://github.com/user-attachments/assets/2c3ff3d2-5e8d-4f76-b42d-5c82b5d812a9)

    **<Terms 변수 분석>**
    
    ```python
    # terms를 순서가 있는 범주형으로 변환 (7, 15, 30 순서로)
    new_df['terms'] = pd.Categorical(new_df['terms'], categories=['7', '15', '30'], ordered=True)
    
    # 첫 번째 플롯: 빈도수 막대 그래프
    sns.countplot(x='terms', hue='loan_status', data=new_df, palette='Set1')
    
    # 두 번째 플롯: 비율 막대 그래프
    sns.histplot(data=new_df, x='terms', hue='loan_status', multiple='fill', shrink=0.5, palette='Set1')
    plt.ylim(0, 1)
    ```
    
     **Terms** 변수는 대출의 상환 기간을 나타내며, 대출 조건에 따라 7일, 15일, 30일의 기간으로 구분됩니다. 이 변수를 순서가 있는 범주형 변수로 변환하여 분석을 진행하였습니다.
    
    - **빈도수 분석**: 상환 기간별 대출 성공 및 실패 빈도수를 시각화하였습니다.
    - **비율 분석**: 각 상환 기간에서 대출 성공과 실패의 비율을 시각화하였습니다.
    
     이를 통해, 대출 상환 기간이 길수록 상환 성공률이 높아지는 경향이 있음을 확인할 수 있었습니다.
    
    ![Figure_2](https://github.com/user-attachments/assets/8589acf7-e3fb-40cd-98da-73f953c36780)

    
    **<Effective Date 변수 분석>**
    
    ```python
    # 첫 번째 플롯: 빈도수 막대 그래프
    sns.histplot(data=new_df, x='effective_date', shrink=0.8, hue='loan_status', multiple='dodge', palette='Set1')
    
    # 두 번째 플롯: 비율 막대 그래프
    sns.histplot(data=new_df, x='effective_date', hue='loan_status', multiple='fill', palette='Set1')
    ```
    
     **Effective Date** 변수는 대출이 시작된 날짜를 나타냅니다. 날짜별 대출 성공 및 실패 빈도수와 비율을 분석하여 특정 기간 동안의 상환 성공률 변화를 살펴보았습니다.
    
    - **빈도수 분석**: 대출이 시작된 날짜에 따른 대출 성공 및 실패 빈도수를 시각화하였습니다.
    - **비율 분석**: 각 날짜에서 대출 성공과 실패의 비율을 시각화하였습니다.
    
     이 분석을 통해 특정 날짜에 대출 상환 성공률이 높아지는 패턴을 확인할 수 있었습니다.
    
    ![Figure_3](https://github.com/user-attachments/assets/95d25b22-8a9e-43ef-be89-de19bb528bfb)

    
    **<Education 변수 분석>**
    
    ```python
    # education을 순서가 있는 범주형으로 변환
    new_df['education'] = pd.Categorical(new_df['education'], categories=["High School or Below", "college", "Bechalor"], ordered=True)
    
    # 첫 번째 플롯: 빈도수 막대 그래프
    sns.histplot(data=new_df, x='education', hue='loan_status', shrink=0.7, multiple='stack', palette='Set1')
    
    # 두 번째 플롯: 비율 막대 그래프
    sns.histplot(data=new_df, x='education', hue='loan_status', shrink=0.7, multiple='fill', palette='Set1')
    ```
    
     **Education** 변수는 고객의 학력을 나타내며, 분석을 위해 학력을 3가지 범주로 단순화하였습니다: `High School or Below`, `College`, `Bechalor`.
    
    - **빈도수 분석**: 학력 수준에 따른 대출 성공 및 실패 빈도수를 시각화하였습니다.
    - **비율 분석**: 각 학력 수준에서 대출 성공과 실패의 비율을 시각화하였습니다.
    
     분석 결과, 고학력자일수록 대출 상환 성공률이 높다는 결론을 얻을 수 있었습니다.
    
    ![Figure_4](https://github.com/user-attachments/assets/44ed32b7-efe5-4f48-bfdc-d110b9f33f8e)

    
    **<Gender 변수 분석>**
    
    ```python
    # 첫 번째 플롯: 빈도수 막대 그래프
    sns.histplot(data=new_df, x='Gender', hue='loan_status', shrink=0.5, multiple='stack', palette='Set1')
    
    # 두 번째 플롯: 비율 막대 그래프
    sns.histplot(data=new_df, x='Gender', hue='loan_status', shrink=0.5, multiple='fill', palette='Set1')
    ```
    
      **Gender** 변수는 고객의 성별을 나타내며, 성별에 따른 대출 성공 및 실패를 분석하였습니다.
    
    - **빈도수 분석**: 성별에 따른 대출 성공 및 실패 빈도수를 시각화하였습니다.
    - **비율 분석**: 각 성별에서 대출 성공과 실패의 비율을 시각화하였습니다.
    
     이를 통해 성별에 따른 대출 상환 성공률의 차이를 확인할 수 있었습니다.
    
    ![Figure_5](https://github.com/user-attachments/assets/a46eed7d-76c1-4101-8f8a-ee1fa40d96b4)

    
    **<기간 (Effective Date ~ Due Date) 분석>**
    
    ```python
    # 기간 변수 생성
    new_df['period'] = new_df['due_date'].dt.dayofyear - new_df['effective_date'].dt.dayofyear
    
    # 첫 번째 플롯: 빈도수 막대 그래프
    sns.histplot(data=new_df, x='period', hue='loan_status', multiple='dodge', palette='Set1')
    
    # 두 번째 플롯: 비율 막대 그래프
    sns.histplot(data=new_df, x='period', hue='loan_status', multiple='fill', palette='Set1')
    ```
    
     **기간** 변수는 대출 실행일부터 상환 마감일까지의 일수 차이를 나타내며, 기간에 따른 대출 성공 및 실패를 분석하였습니다.
    
    - **빈도수 분석**: 기간에 따른 대출 성공 및 실패 빈도수를 시각화하였습니다.
    - **비율 분석**: 각 기간에서 대출 성공과 실패의 비율을 시각화하였습니다.
    
     이 분석을 통해, 대출 상환 기간이 길어질수록 상환 성공률이 변화하는 패턴을 관찰할 수 있었습니다.
    
    ![Figure_6](https://github.com/user-attachments/assets/c5da58e5-3c23-4d8a-9731-e0508c9eb699)

    
- **3. 데이터 가공**
    
     데이터 분석 결과에 따라 데이터 가공을 시작했습니다.  주어진 데이터셋을 모델 학습에 적합하도록 전처리하고, 이를 두 개의 학습 세트와 하나의 테스트 세트로 나누었습니다.
    
    ---
    
     **<데이터 로드 및 추출>**
    
    ```python
    # Load and preprocess loan data
    data = pd.read_csv(PATH + 'dataset.csv')
    
    # 필요한 칼럼만 추출
    columns_to_extract = ['loan_status', 'Principal', 'terms', 'effective_date', 'due_date', 'age', 'education', 'Gender']
    extracted_data = data[columns_to_extract].copy()
    ```
    
    - 먼저, 데이터셋을 로드하고 분석에 필요한 칼럼만 추출합니다. 여기서는 `loan_status`, `Principal`, `terms`, `effective_date`, `due_date`, `age`, `education`, `Gender` 칼럼만을 추출하여 새로운 데이터프레임 `extracted_data`에 저장합니다.
    
     **<날짜 형식 변환 및 파생 변수 생성>**
    
    ```python
    # 날짜 형식 변환
    extracted_data['effective_date'] = pd.to_datetime(extracted_data['effective_date'], format='%m/%d/%Y')
    
    # 'effective_day' 파생 변수 생성 (요일 정보)
    extracted_data['effective_day'] = extracted_data['effective_date'].dt.strftime('%m%d.%a')
    
    # 'effective_day'를 범주형 변수로 변환
    extracted_data['effective_day'] = extracted_data['effective_day'].astype('category')
    
    # 'due_date'도 날짜 형식으로 변환
    extracted_data['due_date'] = pd.to_datetime(extracted_data['due_date'], format='%m/%d/%Y')
    
    # 'due_month'와 'due_day' 파생 변수 생성
    extracted_data['due_month'] = extracted_data['due_date'].dt.month
    extracted_data['due_day'] = extracted_data['due_date'].dt.dayofweek
    
    # 'due_month'와 'due_day'를 범주형 변수로 변환
    extracted_data['due_month'] = extracted_data['due_month'].astype('category')
    extracted_data['due_day'] = extracted_data['due_day'].astype('category')
    
    # 'period' 파생 변수 생성 (effective_date와 due_date 사이의 일수 차이)
    extracted_data['period'] = extracted_data['due_date'].dt.dayofyear - extracted_data['effective_date'].dt.dayofyear
    ```
    
    - `effective_date`와 `due_date`를 `datetime` 형식(날짜 형식)으로 변환합니다.
    - `effective_day`는 `effective_date`로부터 요일 정보를 추출하여 새로운 범주형 변수로 생성합니다.
    - `due_month`는 상환 마감일의 월을 나타내며, `due_day`는 마감일의 요일 정보를 나타냅니다.
    - `period`는 대출 실행일부터 상환 마감일까지의 일수 차이를 계산하여 새롭게 추가한 변수입니다.
    
     **<`loan_status` 및 `education` 변수 변환>**
    
    ```python
    # loan_status 변환 (PAIDOFF -> success, 나머지 -> fail)
    def transform_loan_status(status):
        if status == 'PAIDOFF':
            return 'success'
        else:
            return 'fail'
    
    # education 변환 (Bechalor 이상 -> Bechalor로 통합)
    def transform_education(status):
        if status == 'Bechalor' or status == 'Master or Above':
            return 'Bechalor'
        else:
            return status
    
    # 변환 적용
    extracted_data['loan_status'] = extracted_data['loan_status'].apply(transform_loan_status)
    extracted_data['education'] = extracted_data['education'].apply(transform_education)
    
    # education과 Gender를 범주형 변수로 변환
    extracted_data.loc[:, 'education'] = extracted_data['education'].astype('category')
    extracted_data.loc[:, 'Gender'] = extracted_data['Gender'].astype('category')
    ```
    
    - `loan_status`는 대출이 성공적으로 상환된 경우(`PAIDOFF`)를 `success`로, 그 외의 경우를 `fail`로 변환합니다.
    - `education`은 `Bechalor`와 `Master or Above`를 `Bechalor`로 통합하여 단순화합니다.
    - 마지막으로, `education`과 `Gender`는 범주형 변수로 변환됩니다.
    
     **<범주형 변수 인코딩>**
    
    ```python
    from sklearn.preprocessing import LabelEncoder
    
    # Encode categorical variables using LabelEncoder
    label_encoder = LabelEncoder()
    extracted_data['effective_day'] = label_encoder.fit_transform(extracted_data['effective_day'])
    extracted_data['loan_status'] = label_encoder.fit_transform(extracted_data['loan_status'])
    extracted_data['education'] = label_encoder.fit_transform(extracted_data['education'])
    extracted_data['Gender'] = label_encoder.fit_transform(extracted_data['Gender'])
    ```
    
    - 이 단계에서 범주형 변수(`effective_day`, `loan_status`, `education`, `Gender`)는 `LabelEncoder`를 사용하여 숫자형으로 변환됩니다. 이 작업은 모델에 입력될 수 있는 형태로 데이터를 변환하는 과정입니다.
    
     **<데이터 분할 및 표준화>**
    
    ```python
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    
    # 수치형 및 범주형 변수 구분
    numeric_features = ['Principal', 'age', 'period']
    categorical_features = ['terms', 'effective_day', 'due_month', 'due_day', 'education', 'Gender']
    
    # ColumnTransformer를 사용하여 수치형 변수는 표준화, 범주형 변수는 원-핫 인코딩 적용
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    # Transform the features using the preprocessor
    X_processed = preprocessor.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.3, random_state=42)
    ```
    
    데이터를 학습과 테스트 세트로 분할하고, 수치형 변수를 표준화합니다.
    
    - `Principal`, `age`, `period` 등 수치형 변수는 `StandardScaler`로 표준화합니다.
    - `terms`, `effective_day`, `due_month`, `due_day`, `education`, `Gender` 등 범주형 변수는 `OneHotEncoder`를 사용하여 원-핫 인코딩됩니다.
    - 전처리된 데이터를 학습 세트와 테스트 세트로 나눕니다.
    
     **<데이터 저장>**
    
    ```python
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
    ```
    
    - 전처리된 데이터를 다시 `DataFrame`으로 변환하고, 학습 세트와 테스트 세트를 CSV 파일로 저장합니다.
    - 학습 세트는 두 개의 그룹으로 나누어 각 그룹을 별도로 저장하여 연방 학습이나 다른 목적에 사용할 수 있게 합니다.
    

### 2. 연방 학습

- **1. 개요**
    1. **연방학습(Federated Learning)의 개념**
        
         연방학습(Federated Learning)은 중앙 서버에 데이터를 모으지 않고, 분산된 여러 클라이언트(예: 스마트폰, 로컬 서버)에서 각자의 데이터를 활용하여 모델을 학습한 뒤, 중앙 서버에서 모델을 집계하는 방식의 기계 학습 기술입니다. 각 클라이언트는 로컬 데이터를 사용해 개별적으로 모델을 학습하고, 학습된 모델의 가중치만 중앙 서버로 전송합니다. 중앙 서버는 이 가중치들을 평균 내거나 합쳐서 글로벌 모델을 업데이트합니다. 이 과정에서 원본 데이터는 클라이언트 외부로 유출되지 않으므로, 데이터 프라이버시를 보호할 수 있는 장점이 있습니다.
        
    2. **프로젝트에서 연방학습을 선택한 이유**
        
         이 프로젝트에서 연방학습을 선택한 이유는 두 가지 주요 목적이 있습니다:
        
        - **데이터 프라이버시 보호**: 금융 데이터를 사용하여 모델을 학습하기 때문에, 고객의 민감한 정보를 안전하게 보호할 필요가 있습니다. 연방학습을 통해 데이터를 클라이언트 로컬 환경에 유지하면서도, 모델을 학습할 수 있었습니다. 이는 데이터가 중앙 서버로 전송되지 않으므로, 데이터 유출의 위험이 크게 줄어듭니다.
        - **분산 데이터의 활용**: 이 프로젝트에서는 여러 클라이언트(예: 지역별 데이터 센터)에 분산된 데이터를 활용하여 모델을 학습했습니다. 개별 클라이언트의 데이터가 적을 때는 단독으로 학습하기 어려운 경우가 많지만, 연방학습을 통해 각 클라이언트의 데이터를 효율적으로 결합하여 글로벌 모델을 학습할 수 있었습니다. 이는 데이터의 양과 다양성을 증가시켜 모델의 성능을 향상시킵니다.
    3. **연방학습의 중요성**
        
         연방학습은 특히 개인 정보 보호가 중요한 분야에서 중요한 기술로 자리잡고 있습니다. 예를 들어, 의료, 금융, 스마트 기기 등의 분야에서 연방학습을 통해 민감한 데이터를 중앙 서버에 저장하지 않으면서도, 다양한 데이터를 활용한 고성능 모델을 구축할 수 있습니다. 본 프로젝트에서 연방학습을 도입함으로써, 실용적인 머신러닝 애플리케이션에서 프라이버시와 성능을 동시에 고려한 학습 환경을 구현했습니다.
        
- **2. 모델 구성**
    - **1. 클라이언트-서버 아키텍처**
        
         이 프로젝트에서는 연방학습(Federated Learning)을 구현하기 위해 클라이언트-서버 구조를 사용하였습니다. 두 개의 피어(peer)로 구성된 네트워크에서 한 피어는 서버 역할을, 다른 피어는 클라이언트 역할을 수행합니다. 피어1(Peer1)은 서버 역할을 맡고 있으며, 피어2(Peer2)는 클라이언트 역할을 수행합니다. 이 구조를 통해 두 피어 간의 모델을 공유하고 협력적으로 학습을 수행할 수 있습니다.
        
         각 피어는 로컬 데이터셋을 사용해 독립적으로 모델을 학습하고, 학습된 모델의 가중치를 서로 교환하여 글로벌 모델을 업데이트합니다. 클라이언트-서버 구조를 사용하여 피어 간의 통신을 TCP/IP 프로토콜로 처리하며, 데이터 전송은 모델의 가중치에 한정되어 있어 데이터 프라이버시를 보호할 수 있습니다.
        
        ---
        
        **<클라이언트-서버 통신 흐름>**
        
        1. **서버(Peer1)와 클라이언트(Peer2)의 초기 설정**
            - 각 피어는 자신의 로컬 환경에서 데이터를 로드하고, 모델을 학습하기 위한 초기 설정을 마칩니다. 이 과정에서 클라이언트와 서버는 `socket` 모듈을 사용하여 TCP 소켓 통신을 설정합니다.
            - 피어1은 서버로서 `socket.bind()`와 `socket.listen()`을 통해 클라이언트로부터의 연결을 대기합니다.
            - 피어2는 클라이언트로서 서버에 연결을 시도하며, 연결이 성공하면 서버로부터 모델을 수신할 준비를 합니다.
        2. **로컬 모델 학습 및 전송 (서버-클라이언트)**
            - 피어1(Peer1)은 자신의 로컬 데이터셋을 사용하여 첫 번째 라운드의 모델을 학습합니다. 학습이 완료되면, 학습된 모델을 `.pt` 형식의 파일로 저장합니다.
            - 피어1은 학습된 모델을 피어2(Peer2)로 전송합니다. 이때, 모델 파일의 이름, 크기 등의 메타데이터가 먼저 전송되고, 이후에 모델 파일이 전송됩니다. 전송은 `peer1_send_mdl()` 함수에서 처리됩니다.
            - 피어2는 피어1로부터 모델을 수신받고, 로컬에 저장합니다. 이 과정은 `peer2_recv_mdl()` 함수에서 처리되며, 수신된 모델은 다음 학습 라운드에서 사용됩니다.
        3. **모델 수신 및 로컬 학습 (클라이언트-서버)**
            - 피어2는 피어1로부터 수신한 모델을 사용하여 자신의 로컬 데이터셋으로 학습을 진행합니다. 학습이 완료된 모델은 `.pt` 형식의 파일로 저장됩니다.
            - 피어2는 학습이 완료된 모델을 피어1로 다시 전송합니다. 이때도 마찬가지로 `peer2_send_mdl()` 함수를 사용하여 모델 파일을 전송합니다.
            - 피어1은 피어2로부터 수신한 모델을 받아 로컬에 저장하며, 이 모델을 사용하여 새로운 글로벌 모델을 생성하기 위한 가중치 병합을 수행합니다.
        4. **모델 병합 및 평가**
            - 피어1은 피어2로부터 수신한 모델과 자신의 모델을 병합하여 새로운 글로벌 모델을 생성합니다. 이 과정은 `avg_mdls()` 함수에서 처리됩니다. 병합된 글로벌 모델은 다음 학습 라운드에서 사용됩니다.
            - 병합된 글로벌 모델은 테스트 데이터셋을 사용하여 성능을 평가하며, 평가 결과는 정확도와 손실 곡선으로 시각화됩니다.
        5. **다음 라운드 준비**
            - 각 라운드가 종료되면, 피어1은 다음 라운드를 준비합니다. 이때, 이전 라운드의 모델을 기반으로 새로운 모델이 생성되며, 필요한 경우 이전 라운드의 모델을 복사하여 새 모델로 사용합니다.
            - 이 과정에서 피어1과 피어2는 계속해서 통신을 유지하며, 각 라운드마다 모델을 주고받아 연방학습을 진행합니다.
        
        **<코드 분석>**
        
        1. **`peer1_main.py` - 서버 역할을 담당하는 피어1**
            
             피어1은 서버 역할을 담당하며, 로컬에서 모델을 학습한 후 그 모델을 피어2로 전송합니다. 이후 피어2로부터 수신된 모델을 받아 글로벌 모델을 업데이트하고 평가합니다.
            
            ---
            
            ```python
            # Local Training
            local_model, loss_list, acc_list = \
            local_train_dl(path=PATH, peer_id=peer_id, device=device,
            batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, curr_round=round)
            
            # 학습된 모델을 저장합니다.
            model_save_path = os.path.abspath(os.path.join(PATH, 'peer1', f'peer1_mdl{round}_2.pt'))
            torch.save(local_model.state_dict(), model_save_path)
            ```
            
             이 부분은 피어1에서 로컬 데이터셋을 사용하여 모델을 학습한 후, 해당 모델을 `.pt` 파일로 저장하는 코드입니다. 모델은 이후 피어2로 전송됩니다.
            
            ```python
            # Sending Local Model to Peer2
            input('Send Local trained model to peer2: Press Enter to continue...')
            peer1_send_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR, round=round, path=PATH)
            ```
            
             이 코드는 학습된 모델을 피어2로 전송하는 부분입니다. `peer1_send_mdl()` 함수가 호출되며, TCP 소켓을 통해 피어2에 연결하고 모델 파일을 전송합니다.
            
            ```python
            # Receiving Peer2 Model
            input("Receive model from peer2: Press Enter to continue...")
            connected = peer1_recv_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR, path=PATH)
            ```
            
             이 부분에서는 피어2로부터 모델을 수신합니다. `peer1_recv_mdl()` 함수가 호출되어 피어2로부터 전송된 모델 파일을 로컬에 저장합니다.
            
            ```python
            # Averaging Peer1 & Peer2 models
            if connected:
                new_mdl = avg_mdls(peer_id=peer_id, round=round, path=PATH)
                torch.save(new_mdl.state_dict(), PATH + f"peer1\peer1_mdl{round}_1.pt")
            ```
            
             이 코드는 피어1과 피어2의 모델을 평균하여 새로운 글로벌 모델을 생성하는 부분입니다. `avg_mdls()` 함수가 호출되어 모델의 가중치를 병합하고, 병합된 모델을 새로운 글로벌 모델로 저장합니다.
            
        2. **`peer2_main.py` - 클라이언트 역할을 담당하는 피어2**
            
             피어2는 클라이언트 역할을 담당하며, 피어1로부터 모델을 수신하여 자신의 로컬 데이터로 학습을 진행합니다. 학습이 완료된 모델을 다시 피어1로 전송합니다.
            
            ---
            
            ```python
            # Receiving Peer1 Model, current round
            input("\n\nReceive model from peer1: Press Enter to continue...")
            curr_round = peer2_recv_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR, path=PATH)
            ```
            
             이 코드는 피어1로부터 현재 라운드의 모델을 수신받는 부분입니다. `peer2_recv_mdl()` 함수가 호출되어 피어1에서 전송된 모델을 로컬에 저장합니다.
            
            ```python
            # Local Training
            local_model, loss_list, acc_list = \
            local_train_dl(path=PATH, peer_id=peer_id, device=device,
            batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, curr_round=curr_round)
            
            # 학습된 모델을 저장합니다.
            torch.save(local_model.state_dict(), f'peer2\peer2_mdl{curr_round}_2.pt')
            ```
            
             이 부분은 피어2에서 수신된 모델을 사용하여 로컬 데이터셋으로 학습을 진행하고, 학습된 모델을 `.pt` 파일로 저장하는 코드입니다. 이 모델은 이후 피어1으로 전송됩니다.
            
            ```python
            # Sending Local Model to Peer1
            input('Send Local trained model to peer1: Press Enter to continue...')
            peer2_send_mdl(host=HOST, port=PORT, bufsize=BUFFER_SIZE, sep=SEPARATOR, round=curr_round, path=PATH)
            ```
            
             이 코드는 학습된 모델을 피어1로 전송하는 부분입니다. `peer2_send_mdl()` 함수가 호출되어 피어1에 모델 파일을 전송합니다.
            
        3. **`p2p_comm.py` - 피어 간의 통신 처리**
            
            이 파일은 피어1과 피어2 간의 TCP 소켓 통신을 처리하는 함수들을 포함합니다.
            
            ---
            
            ```python
            # Sending current round, peer1 model
            def peer1_send_mdl(host, port, bufsize, sep, round, path):
                def send_model():
                    try:
                        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        serverSocket.bind((host, port))
                        serverSocket.listen(1)
                        conn, addr = serverSocket.accept()
            
                        filename = path + f'peer1\peer1_mdl{round}_2.pt'
                        filesize = os.path.getsize(filename)
                        conn.sendall(f"{round}{sep}{filename}{sep}{filesize}".encode())
            
                        with open(filename, "rb") as f:
                            while True:
                                bytes_read = f.read(bufsize)
                                if not bytes_read:
                                    break
                                conn.sendall(bytes_read)
                    finally:
                        conn.close()
                        serverSocket.close()
                
                t = threading.Thread(target=send_model)
                t.start()
            ```
            
             이 함수는 피어1에서 피어2로 모델을 전송하는 역할을 합니다. 소켓을 설정하고, 전송할 모델 파일의 메타데이터(라운드 번호, 파일 크기 등)를 전송한 후, 파일 데이터를 전송합니다. 스레드를 사용하여 비동기적으로 모델 전송을 처리합니다.
            
            ```python
            # Receiving model from peer2
            def peer1_recv_mdl(host, port, bufsize, sep, path):
                try:
                    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    serverSocket.bind((host, port))
                    serverSocket.listen(1)
                    conn, addr = serverSocket.accept()
                    
                    received = conn.recv(bufsize).decode().split(sep)
                    curr_round = int(received[0])
                    filename = os.path.basename(received[1])
                    filesize = int(received[2])
            
                    with open(path + r"peer1\recvd_models\\" + filename, "wb") as f:
                        total_received = 0
                        while total_received < filesize:
                            bytes_read = conn.recv(bufsize)
                            if not bytes_read:
                                break
                            f.write(bytes_read)
                            total_received += len(bytes_read)
                finally:
                    conn.close()
                    serverSocket.close()
                return curr_round
            ```
            
             이 함수는 피어2에서 피어1로 모델을 수신하는 역할을 합니다. 소켓을 설정하고, 수신한 모델 파일을 로컬에 저장합니다. 수신된 파일은 이후 글로벌 모델 업데이트에 사용됩니다.
            
        
    - **2. 모델 아키텍처**
        
         이 프로젝트에서는 대출 상환 여부를 예측하기 위해 딥러닝 기반의 인공 신경망(Artificial Neural Network, ANN)을 설계하고 학습시켰습니다. 모델은 연속적인 전결합 층(Fully Connected Layers)과 배치 정규화(Batch Normalization), 드롭아웃(Dropout) 기법을 적용하여 학습 과정의 안정성과 과적합(Overfitting)을 방지하였습니다.
        
        ---
        
        **<모델 구조>**
        
         모델은 총 7개의 전결합 층으로 구성되어 있으며, 입력 특성(Input Features)에서부터 최종 출력층(Output Layer)까지의 정보 흐름을 처리합니다. 각 층은 배치 정규화 및 드롭아웃 레이어를 포함하고 있어 학습 과정에서의 안정성과 성능을 높이도록 설계되었습니다.
        
        ```python
        class LoanNet(nn.Module):
            def __init__(self, input_dim):
                super(LoanNet, self).__init__()
                self.fc1 = nn.Linear(input_dim, 1024)
                self.fc2 = nn.Linear(1024, 512)
                self.fc3 = nn.Linear(512, 256)
                self.fc4 = nn.Linear(256, 128)
                self.fc5 = nn.Linear(128, 64)
                self.fc6 = nn.Linear(64, 32)
                self.fc7 = nn.Linear(32, 2)
                self.dropout = nn.Dropout(0.35)  # 드롭아웃 비율 변경
                self.batch_norm1 = nn.BatchNorm1d(1024)
                self.batch_norm2 = nn.BatchNorm1d(512)
                self.batch_norm3 = nn.BatchNorm1d(256)
                self.batch_norm4 = nn.BatchNorm1d(128)
                self.batch_norm5 = nn.BatchNorm1d(64)
                self.batch_norm6 = nn.BatchNorm1d(32)
        
            def forward(self, x):
                x = F.relu(self.batch_norm1(self.fc1(x)))
                x = self.dropout(x)
                x = F.relu(self.batch_norm2(self.fc2(x)))
                x = self.dropout(x)
                x = F.relu(self.batch_norm3(self.fc3(x)))
                x = self.dropout(x)
                x = F.relu(self.batch_norm4(self.fc4(x)))
                x = self.dropout(x)
                x = F.relu(self.batch_norm5(self.fc5(x)))
                x = self.dropout(x)
                x = F.relu(self.batch_norm6(self.fc6(x)))
                x = self.fc7(x)
                return x
        ```
        
        모델의 주요 구성 요소는 다음과 같습니다:
        
        - **전결합층 (Fully Connected Layer)**: 각 레이어는 입력으로 받은 데이터에 선형 변환을 수행하며, 이 때 가중치와 편향이 적용됩니다. 입력 데이터의 차원을 1024, 512, 256, 128, 64, 32로 점진적으로 축소하면서 중요한 특징을 추출합니다.
        - **활성화 함수 (Activation Function)**: 각 레이어에서 비선형성을 추가하기 위해 ReLU(Rectified Linear Unit) 함수가 사용됩니다. 이는 모델이 복잡한 비선형 관계를 학습할 수 있도록 도와줍니다.
        - **배치 정규화 (Batch Normalization)**: 각 레이어 뒤에 배치 정규화가 적용되어, 학습 과정에서 미니 배치의 출력이 정규화되도록 합니다. 이는 학습 속도를 높이고, 내부 공변량 변화(Internal Covariate Shift)를 줄여 모델의 성능을 개선합니다.
        - **드롭아웃 (Dropout)**: 드롭아웃은 과적합을 방지하기 위해 일부 뉴런을 임의로 제거하는 기법입니다. 각 레이어 뒤에 35%의 드롭아웃 비율이 적용되어, 모델의 일반화 능력을 향상시킵니다.
        
        **<학습 과정>**
        
         모델은 `CrossEntropyLoss`를 손실 함수로 사용하여 예측과 실제 값 간의 차이를 최소화하도록 학습됩니다. 옵티마이저로는 Adam 옵티마이저가 사용되었으며, 학습 속도를 동적으로 조정하기 위해 ReduceLROnPlateau 스케줄러가 적용되었습니다.
        
        ```python
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        ```
        
         이러한 설정을 통해 모델은 각 라운드에서 학습을 진행하며, 로컬 데이터로 학습된 모델을 피어 간에 교환하고 평균화하여 글로벌 모델로 업데이트합니다. 이 과정을 통해 분산된 데이터를 활용한 연방 학습이 이루어집니다.
        
        **<모델 평가>**
        
         학습이 완료된 후, 테스트 데이터셋을 사용하여 모델의 성능을 평가합니다. 모델의 손실(Loss)과 정확도(Accuracy)가 주요 성능 지표로 사용되며, 각 라운드마다 모델의 성능 변화를 확인할 수 있습니다.
        
        ```python
        loss, acc = evaluate_dl(device=device, model=new_model, criterion=criterion, test_loader=test_loader)
        print(f'>>> Round {round} - Test Loss: {loss:.4f}  Test Accuracy: {acc:.2f}%')
        ```
        
         이 과정을 통해 학습된 모델이 새로운 데이터에 대해 얼마나 잘 예측할 수 있는지 평가합니다. 테스트 정확도가 지속적으로 증가하는지, 혹은 어느 정도 수렴하는지를 통해 모델의 최종 성능을 판단할 수 있습니다.
        

## **3. 결과 분석**

 이번 실험에서는 연방학습과 전체 데이터셋을 사용한 학습 간의 성능 차이를 비교했습니다. 연방학습에서는 데이터를 두 개의 피어(peer)로 나누어 각 피어에서 모델을 학습한 후, 이들 모델을 통합하여 최종 모델을 생성했습니다. 이를 통해 연방학습이 데이터 분할에도 불구하고 성능 향상을 가져오는지 확인하고자 했습니다.

---

### **1. 연방학습 결과**

![Figure_1](https://github.com/user-attachments/assets/2d93a01e-7ab9-4be9-9f66-7ab2b73bfc92)

 연방학습을 적용한 결과, 각 라운드(round)별 테스트 손실(Test Loss)과 테스트 정확도(Test Accuracy)의 변화를 관찰할 수 있었습니다. 위 이미지는 각각 연방학습 중 라운드별 테스트 손실과 테스트 정확도의 변화를 시각화한 것입니다.

- **테스트 손실 (Test Loss)**: 초기 라운드에서 테스트 손실이 점진적으로 감소하다가, 후반 라운드에서는 다소 증가하는 경향을 보였습니다. 이는 연방학습 과정에서 모델 간의 조정이 이루어지며 발생하는 일시적인 손실 변동일 수 있습니다.
- **테스트 정확도 (Test Accuracy)**: 반면 테스트 정확도는 연방학습이 진행됨에 따라 꾸준히 증가하는 양상을 보였습니다. 특히 마지막 라운드에서는 약 **62.67%**의 정확도를 기록하며, 학습이 진행될수록 모델의 성능이 향상됨을 확인할 수 있었습니다.

### 2. 전체 데이터셋을 사용한 학습 결과와의 비교

 연방학습의 결과를 전체 데이터셋을 사용한 학습 결과와 비교해보면, 연방학습을 적용하지 않았을 때 모델의 테스트 정확도는 55.33%에 그쳤습니다. 이는 연방학습을 통해 얻어진 최종 모델의 정확도인 62.67%보다 약 **7.34%** 낮은 수치입니다.

### 3. 결과 해석

 이러한 결과는 연방학습의 효과를 잘 보여줍니다. 연방학습은 데이터의 분할에도 불구하고 모델 성능을 향상시키는 데 기여할 수 있음을 시사합니다. 특히, 데이터의 다양성을 증가시키고, 각 피어에서 학습된 모델을 통합함으로써 모델의 일반화 성능이 개선되었음을 확인할 수 있습니다.

 또한, 연방학습이 적용된 환경에서는 데이터의 지역적 편향을 줄일 수 있어, 보다 균형 잡힌 모델을 구축할 수 있음을 알 수 있습니다. 이는 특히 데이터 프라이버시 보호가 중요한 상황에서 연방학습의 강력한 장점을 부각시키는 부분입니다.

 결론적으로, 이번 실험을 통해 연방학습이 데이터의 분할과 제한된 학습 자원에도 불구하고, 모델 성능을 효과적으로 개선할 수 있음을 확인할 수 있었습니다. 이는 향후 다양한 실세계 응용에서 연방학습을 고려할 충분한 이유가 될 수 있습니다.
