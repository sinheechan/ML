---
layout: single
title:  "jupyter notebook 변환하기!"
categories: coding
tag: [python, blog, jupyter]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# [Dataest] Titanic 생존률 예측하기


- 1912년도 4월 15일 타이타닉호가 빙산에 충돌하여 침몰하였습니다.

- 이는 탑승한 사람들을 위한 구명보트가 충분하지 않아 승객과 승무원 2,224명 중 1,502명이 사망하였습니다

- 생존에는 어느 정도 운이 관련되어 있으나, 일부 집단 사람들은 다른 집단보다 생존 가능성이 더 높았던 것으로 추측됩니다.

- 이에 본 분석에서는 "어떤 종류의 사람들이 생존 할 가능성이 더 높은가?"라는 질문에 대한 예측모델을 구축하는 것을 목표로 합니다.


## 개요


- 본 분석에 대한 자료는 캐글에서 제공하는 (Dataset) Titanic - Machine Learning from Disaster 에서 다운받을 수 있습니다.

- 이 자료는 탑승객의 정보를 포함하는 두 개의 유사한 데이터 세트를 다운받을 수 있습니다.

  - Train.csv : 탑승한 승객 중 891명에 대한 세부정보가 포함되며 이 탑승객에 대한 실측 진실이 공개됩니다.

  - test.csv : 탑승객 418명의 세부정보가 들어있으나 탑승객에 대한 실측 진실이 공개되지 않습니다.

- 따라서 위 자료 특성 및 패턴에 따라 train.csv탑승한 다른 418명의 승객의 생존여부를 예측하는 것을 중점으로 분석을 시행합니다.


## 1. 초기 세팅


#### 1.1 라이브러리 Import



```python
import pandas as pd
import numpy as np

np.random.seed(2024)
np.set_printoptions(precision = 4, suppress = True)

import matplotlib.pyplot as plt
import seaborn as sns

PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
pd.options.display.max_columns = 20
pd.options.display.max_colwidth = 80

plt.rc("figure", figsize = (8, 4))

sns.set_theme()

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
```

#### 1.2 폰트 설정



```python
import matplotlib.font_manager as fm
import os
import matplotlib.pyplot as plt

!apt -qq install fonts-nanum

fe = fm.FontEntry(
    fname = r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
    name = 'NanumGothic')
fm.fontManager.ttflist.insert(0, fe)
plt.rcParams.update({'font.size': 12, 'font.family': 'NanumGothic'})
```

<pre>
The following NEW packages will be installed:
  fonts-nanum
0 upgraded, 1 newly installed, 0 to remove and 24 not upgraded.
Need to get 10.3 MB of archives.
After this operation, 34.1 MB of additional disk space will be used.
Selecting previously unselected package fonts-nanum.
(Reading database ... 121654 files and directories currently installed.)
Preparing to unpack .../fonts-nanum_20200506-1_all.deb ...
Unpacking fonts-nanum (20200506-1) ...
Setting up fonts-nanum (20200506-1) ...
Processing triggers for fontconfig (2.13.1-4.2ubuntu5) ...
</pre>
## 2. 데이터 불러오기



```python
train = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")

all_data = [train,test]
```

## 3. EDA



[ train.csv ]

- 총 891개의 데이터 값과 12개의 칼럼을 보유



[ test.csv ]

- 총 418개의 데이터 값과 11개의 칼럼을 보유

- train 데이터와 모든 칼럼이 같으나 survived(생존여부)에 관한 데이터 없음



[ Column 정보 ]

- PassengerId : 탑승객의 ID(인덱스와 같은 개념)

- Survived : 생존유무(0은 사망 1은 생존)

- Pclass : 객실의 등급

- Name : 이름

- Sex :성별

- SibSp : 동승한 형제 혹은 배우자의 수

- Parch : 동승한 자녀 혹은 부모의 수

- Ticket : 티켓번호

- Fare : 요금

- Cabin : 선실

- Embarked : 탑승지 (C = Cherbourg, Q = Queenstown, S = Southampton)




```python
train.head()
```

<pre>
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                  Name     Sex   Age  SibSp  \
0                              Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Thayer)  female  38.0      1   
2                               Heikkinen, Miss. Laina  female  26.0      0   
3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                             Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S  
</pre>

```python
test.head()
```

<pre>
   PassengerId  Pclass                                          Name     Sex  \
0          892       3                              Kelly, Mr. James    male   
1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   
2          894       2                     Myles, Mr. Thomas Francis    male   
3          895       3                              Wirz, Mr. Albert    male   
4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   

    Age  SibSp  Parch   Ticket     Fare Cabin Embarked  
0  34.5      0      0   330911   7.8292   NaN        Q  
1  47.0      1      0   363272   7.0000   NaN        S  
2  62.0      0      0   240276   9.6875   NaN        Q  
3  27.0      0      0   315154   8.6625   NaN        S  
4  22.0      1      1  3101298  12.2875   NaN        S  
</pre>

```python
print(train.info())
print("-"*100)
print(test.info())
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None
----------------------------------------------------------------------------------------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  418 non-null    int64  
 1   Pclass       418 non-null    int64  
 2   Name         418 non-null    object 
 3   Sex          418 non-null    object 
 4   Age          332 non-null    float64
 5   SibSp        418 non-null    int64  
 6   Parch        418 non-null    int64  
 7   Ticket       418 non-null    object 
 8   Fare         417 non-null    float64
 9   Cabin        91 non-null     object 
 10  Embarked     418 non-null    object 
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
None
</pre>

```python
train.isnull().sum()

print(train.isnull().sum())
print("-"*100)
print(test.isnull().sum())
```

<pre>
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
----------------------------------------------------------------------------------------------------
PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
</pre>
## 4. 데이터 전처리


#### 4.1 불필요한 데이터 Drop



- 분석에 불필요하다고 판단되는 데이터 PassengerId, Name, Ticket 칼럼을 Drop한다.

- test 데이터에서는 PassengerId 칼럼이 없으므로 Name, Ticket 칼럼을 Drop 한다.



```python
train = train.drop(["PassengerId", 'Name', 'Ticket'], axis = 1 )
test = test.drop(['Name', 'Ticket'], axis= 1)
```

#### 4.2 데이터 전처리



[ Cabin ]

- NA 값을 문자 N으로 대체한다.

- 전체 값을 앞글자만 딴 이름으로 변경한다.



[ Age ]

- NA 값을 평균값으로 대체한다.



[ Fare ]

- NA 값을 평균값으로 대체한다.



[ Embarked ]

- NA값을 최빈값으로 대체한다.



```python
train['Cabin'].fillna('N', inplace=True)

train['Cabin'] = train['Cabin'].str[:1]
```


```python
train['Age'].fillna(train['Age'].mean(), inplace=True)
```


```python
train["Fare"].fillna(0, inplace=True)
```


```python
train['Embarked'].fillna(train["Embarked"].mode()[0], inplace=True)
```


```python
train.head()
```

<pre>
   Survived  Pclass     Sex   Age  SibSp  Parch     Fare Cabin Embarked
0         0       3    male  22.0      1      0   7.2500     N        S
1         1       1  female  38.0      1      0  71.2833     C        C
2         1       3  female  26.0      0      0   7.9250     N        S
3         1       1  female  35.0      1      0  53.1000     C        S
4         0       3    male  35.0      0      0   8.0500     N        S
</pre>
#### 4.7 데이터 인코딩



- 숫자형 값을 가지지 않은 'Cabin', 'Sex', 'Embarked' 칼럼을 인코딩한다.



```python
from sklearn import preprocessing

def encode_features(dataDF):
  features = ['Cabin', 'Sex', 'Embarked']
  for feature in features:
    le = preprocessing.LabelEncoder()
    le = le.fit(dataDF[feature])
    dataDF[feature] = le.transform(dataDF[feature])
  return dataDF
```


```python
train = encode_features(train)

train.head()
```

<pre>
   Survived  Pclass  Sex   Age  SibSp  Parch     Fare  Cabin  Embarked
0         0       3    1  22.0      1      0   7.2500      7         2
1         1       1    0  38.0      1      0  71.2833      2         0
2         1       3    0  26.0      0      0   7.9250      7         2
3         1       1    0  35.0      1      0  53.1000      2         2
4         0       3    1  35.0      0      0   8.0500      7         2
</pre>
#### 4.7 데이터 전처리 함수화



- fillna : 결측값 대체 및 조정 함수

- drop_features : 불필요한 칼럼 제거함수

- format_features : 데이터 값 요약 및 인코딩 함수

- transform_features : 위 3개 함수를 모두 실행시키는 함수



```python
from sklearn.preprocessing import LabelEncoder

def fillna(df):
  df['Cabin'].fillna('N', inplace=True)
  df['Age'].fillna(df['Age'].mean(), inplace=True) # 평균치로 결측치 채움
  df["Fare"].fillna(df["Fare"].mean(), inplace = True)
  df['Embarked'].fillna(df["Embarked"].mode()[0], inplace=True)
  print('데이터셋 null 개수:', df.isnull().sum().sum())
  return df

def drop_features(df):
  df.drop(['Name', 'Ticket'], axis=1, inplace=True) # 승객아이디보류
  return df

def format_features(df):
  df['Cabin'] = df['Cabin'].str[:1]
  features = ['Cabin', 'Sex', 'Embarked']
  for feature in features:
    le = LabelEncoder()
    le = le.fit(df[feature])
    df[feature] = le.transform(df[feature])
  return df

def transform_features(df):
  df = fillna(df)
  df = drop_features(df)
  df = format_features(df)
  return df
```


```python
df = pd.read_csv('/content/train.csv')

X_df = df.drop('Survived', axis=1)
X_df = transform_features(X_df)

y_df = df['Survived']
```

<pre>
데이터셋 null 개수: 0
</pre>

```python
X_df.head()
```

<pre>
   PassengerId  Pclass  Sex   Age  SibSp  Parch     Fare  Cabin  Embarked
0            1       3    1  22.0      1      0   7.2500      7         2
1            2       1    0  38.0      1      0  71.2833      2         0
2            3       3    0  26.0      0      0   7.9250      7         2
3            4       1    0  35.0      1      0  53.1000      2         2
4            5       3    1  35.0      0      0   8.0500      7         2
</pre>
## 5. 머신러닝


#### 5.1 Test set / Train set 분리



- 비율은 8:2로 설정한다.

- X_train / X_test : 생존 칼럼을 제외한 나머지 데이터 8:2

- y_train / y_test : 생존 칼럼만 추출한 데이터 8:2



```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=2024)
print(len(X_train), len(y_train))
print(len(X_test), len(y_test))
```

<pre>
712 712
179 179
</pre>
#### 5.2 모델 정확도 측정



[ 활용 모델 ]



- dt_clf : 의사결정나무

- rf_clf : 랜덤 포레스트

- lr_clf : 로지스틱 회귀분석



```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier(random_state=2024)
rf_clf = RandomForestClassifier(random_state=2024)
lr_clf = LogisticRegression(solver='liblinear')

dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
print('DT 정확도: {0:.4f}'.format(accuracy_score(y_test, dt_pred)))

rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
print('RF 정확도: {0:.4f}'.format(accuracy_score(y_test, rf_pred)))

lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
print('LR 정확도: {0:.4f}'.format(accuracy_score(y_test, lr_pred)))
```

<pre>
DT 정확도: 0.7654
RF 정확도: 0.8492
LR 정확도: 0.8101
</pre>
#### 5.3 교차검증 ( K-Fold )



- k-fold 갯수 : 5




```python
from sklearn.model_selection import KFold

def exec_kfold(clf, folds=5):
  kfold = KFold(n_splits=folds)
  scores = []
  for i, (train_index, test_index) in enumerate(kfold.split(X_df)):
    X_train, X_test = X_df.values[train_index], X_df.values[test_index]
    y_train, y_test = y_df.values[train_index], y_df.values[test_index]

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    scores.append(accuracy)
    print('교차 검증{0} 정확도:{1:4f}'.format(i, accuracy))

  mean_score = np.mean(scores)
  print('평균정확도:{0:4f}'.format(mean_score))


exec_kfold(dt_clf, folds=5)
```

<pre>
교차 검증0 정확도:0.720670
교차 검증1 정확도:0.808989
교차 검증2 정확도:0.808989
교차 검증3 정확도:0.741573
교차 검증4 정확도:0.393258
평균정확도:0.694696
</pre>

```python
from sklearn.model_selection import cross_val_score, cross_val_predict

cv = KFold(n_splits=5, random_state=2024, shuffle=True)
accs = cross_val_score(dt_clf, X_df, y_df, cv=cv)
print(accs)
print('평균정확도:', np.mean(accs))
```

<pre>
[0.7654 0.7528 0.8258 0.7303 0.7528]
평균정확도: 0.7654321762601217
</pre>
#### 5.4 최적 파라미터 탐색



- 그리드 서치(Grid Search)를 사용하여 의사결정 트리 모델의 최적 하이퍼파라미터를 찾는다.

- 그리고 그에 대한 모델을 훈련하고 테스트 세트에서의 정확도를 출력하는 작업을 진행한다.




```python
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[2,3,5,10],
              'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]} # 높이면 덜 분할 일반화, 높이면 더 작은 리프노드, 간단해짐

grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5) # n_jobs=-1(모든 cpu 사용), verbose=2(로그출력)
grid_dclf.fit(X_train, y_train)

print('GS 최적 하이퍼 파라미터:', grid_dclf.best_params_)
print('GS 최고 정확도:{0:4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_

dpred = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test, dpred)
print('테스트셋 DR 정확도(GS를 한것):{0: 4f}'.format(accuracy))
```

<pre>
GS 최적 하이퍼 파라미터: {'max_depth': 5, 'min_samples_leaf': 5, 'min_samples_split': 2}
GS 최고 정확도:0.804777
테스트셋 DR 정확도(GS를 한것): 0.776536
</pre>
## 6. 제출 파일 형식



-  PassengerId(승객 ID)와 Survived(생존 여부)  두 개의 column으로 이루어진 csv 파일 제출

  - 418개의 항목과 헤더 행이 포함된 csv 파일을 제출

  - test data 내에 PassengerId(승객 ID)

  - test data 에 modeling을 적용하여 얻은‘Survived(생존 여부) 값



```python
test_data = pd.read_csv('/content/test.csv')
#y_test = df['Survived']
#X_test = df.drop('Survived', axis=1)
X_test = transform_features(test_data)
```

<pre>
데이터셋 null 개수: 0
</pre>

```python
test_prediction = best_dclf.predict(X_test)
```


```python
len(test_prediction)
```

<pre>
418
</pre>

```python
submission_df = pd.DataFrame({'PassengerId': X_test['PassengerId'], 'Survived': test_prediction})
submission_df.to_csv('submission.csv', index=False)
```
