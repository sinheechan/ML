

# [Dataest] Titanic 생존률 예측하기

<br/>


- 1912년도 4월 15일 타이타닉호가 빙산에 충돌하여 침몰하였습니다.

- 이는 탑승한 사람들을 위한 구명보트가 충분하지 않아 승객과 승무원 2,224명 중 1,502명이 사망하였습니다

- 생존에는 어느 정도 운이 관련되어 있으나, 일부 집단 사람들은 다른 집단보다 생존 가능성이 더 높았던 것으로 추측됩니다.

- 이에 본 분석에서는 "어떤 종류의 사람들이 생존 할 가능성이 더 높은가?"라는 질문에 대한 예측모델을 구축하는 것을 목표로 합니다.

  <br/>

  <br/>

## 개요

<br/>


- 본 분석에 대한 자료는 캐글에서 제공하는 (Dataset) Titanic - Machine Learning from Disaster 에서 다운받을 수 있습니다.

- 이 자료는 탑승객의 정보를 포함하는 두 개의 유사한 데이터 세트를 다운받을 수 있습니다.

  - Train.csv : 탑승한 승객 중 891명에 대한 세부정보가 포함되며 이 탑승객에 대한 실측 진실이 공개됩니다.

  - test.csv : 탑승객 418명의 세부정보가 들어있으나 탑승객에 대한 실측 진실이 공개되지 않습니다.

- 따라서 위 자료 특성 및 패턴에 따라 train.csv탑승한 다른 418명의 승객의 생존여부를 예측하는 것을 중점으로 분석을 시행합니다.

  <br/>

  <br/>

## 1. 초기 세팅

<br/>


#### 1.1 라이브러리 Import

<br/>

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

<br/>

#### 1.2 폰트 설정

<br/>

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

<br/>

<br/>

## 2. 데이터 불러오기

<br/>

```python
train = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")

all_data = [train,test]
```

<br/>

<br/>

## 3. EDA

<br/>

- train.csv : 총 891개의 데이터 값과 12개의 칼럼을 보유

- test.csv : 총 418개의 데이터 값과 11개의 칼럼을 보유, train 데이터와 모든 칼럼이 같으나 survived(생존여부)에 관한 데이터 없음

<br/>

- Column 정보

  | Columns     | Detail                                                  |
  | ----------- | ------------------------------------------------------- |
  | PassengerId | 탑승객의 ID(인덱스와 같은 개념)                         |
  | Survived    | 생존유무(0은 사망 1은 생존)                             |
  | Pclass      | 객실의 등급                                             |
  | Name        | 이름                                                    |
  | Sex         | 성별                                                    |
  | SibSp       | 동승한 형제 혹은 배우자의 수                            |
  | Parch       | 동승한 자녀 혹은 부모의 수                              |
  | Ticket      | 티켓번호                                                |
  | Fare        | 티켓번호                                                |
  | Cabin       | 선실                                                    |
  | Embarked    | 탑승지 (C = Cherbourg, Q = Queenstown, S = Southampton) |

  <br/>


```python
train.head()
```

<br/>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
<br/>

```python
test.head()
```

<br/>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
<br/>

```python
print(train.info())
print("-"*100)
print(test.info())
```

<br/>

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
<br/>


```python
train.isnull().sum()

print(train.isnull().sum())
print("-"*100)
print(test.isnull().sum())
```

<br/>

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
<br/>

<br/>

## 4. 데이터 전처리

<br/>


#### 4.1 불필요한 데이터 Drop

<br/>

- 분석에 불필요하다고 판단되는 데이터 PassengerId, Name, Ticket 칼럼을 Drop한다.

- test 데이터에서는 PassengerId 칼럼이 없으므로 Name, Ticket 칼럼을 Drop 한다.

<br/>

```python
train = train.drop(["PassengerId", 'Name', 'Ticket'], axis = 1 )
test = test.drop(['Name', 'Ticket'], axis= 1)
```

<br/>

#### 4.2 데이터 전처리

<br/>

- Cabin : NA 값을 문자 N으로 대체한다, 전체 값을 앞글자만 딴 이름으로 변경한다.
- Age : NA 값을 평균값으로 대체한다.

- Fare : NA 값을 평균값으로 대체한다.

- Embarked : NA값을 최빈값으로 대체한다.

<br/>

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



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>N</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>N</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>C</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>N</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
<br/>


#### 4.7 데이터 인코딩

<br/>

- 숫자형 값을 가지지 않은 'Cabin', 'Sex', 'Embarked' 칼럼을 인코딩한다.

<br/>

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

<br/>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>7</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<br/>

<br/>

## 5. 변수 별 생존비율 시각화

<br/>

#### 5.1 Pclass

<br/>

- 객실 수준이 높을수록 생존확률이 높게 나타났다.

- 이에 pclass는 추후 지도학습에서 분석해볼 만한 가치가 있다.

<br/>

```python
def bar_chart(df):
    survived = train.loc[train["Survived"] == 1, df].value_counts().sort_index()
    dead = train.loc[train["Survived"] == 0, df].value_counts().sort_index()

    data = pd.DataFrame([survived, dead], index=["Survived", "Dead"])
    setting = data.plot(kind="bar", figsize=(8, 4))
    plt.show()

bar_chart("Pclass")
```

<br/>

<img src="data:image/png;base64,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AAmDr/uBNmpqatKiRYvUp08fq0sBACBhEFKBMI0ePVp+v7/FMr/fr/379+ukk07Sn/70J4sqAwAg8XBPKhCmTz75pNUjqAzDUPfu3dW3b98Wr0gFAABtQ0gFolRRUaENGzbI7XaruLhYXbp0sbokAAASBpd+gOO44447tHPnzlbLX3/9dV199dX64IMP9Je//EXXXnutampqLKgQAIDExJVU4Djy8vL0wQcftPgqf9euXRozZoyefPJJ5eXlSZIee+wx1dbWasaMGVaVCgBAQuFKKnAC/3yv6YwZM3Tdddc1B1RJGjdunNavXx/v0gAASFiEVOA4+vbtq+3btzf//NJLL+nrr7/WnXfe2aJdWlqa6uvr410eAAAJi0dQAcdx66236p577tHdd9+t/fv3a86cOZo/f77S0tJatNu3b588Ho9FVQIAkHgIqcBxXHrppc0P63c4HJo7d65GjhzZqt3nn3+un/3sZxZUCABAYmLiFAAAAGyHe1IBAABgO4RUAAAA2A4hFQAAALZDSAUAAIDtEFIBAABgO4RUAAAA2A4hFQAAALZDSAUAAIDt/H99eGRZP74rYAAAAABJRU5ErkJggg=="/>

<br/>

#### 5.2 Fare

<br/>

- 요금이 낮을수록 사망자가 높다

- 요금에 따른 사망, 생존확률이 극명하게 구분되어 분석이 필요해보인다.

<br/>

```python
snake = sns.FacetGrid(train, hue="Survived", aspect=4)
snake.map(sns.kdeplot, "Fare", shade=True)
snake.set(xlim=(0, train["Fare"].max()))
snake.add_legend()

plt.xlim(0, 150)
plt.show()
```

<br/>

<img src="data:image/png;base64,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mZld9S0gIiIiIiIiIiLyIcjNrRIRBFu3bsXSpUtRUFCA8PBwTJ06FVOnTgUA1NfXY/z48Vi+fDkGDhzo2Wft2rV46aWXUFxcjNjYWMyePRsTJ070Ou4vv/yCRYsW4eDBg9BoNLj55pvxwAMPQJIkT52ysjIsWrQIW7ZsgSzLGD16NBYsWICoqKiuuXgiIiIiIiIiIiI/QirAIyIiIiIiIiIiIm8hcwstERERERERERER+WKAR0REREREREREFMIY4BEREREREREREYUwBnhEREREREREREQhjAEeERERERERERFRCGOAR0REREREREREFMIY4HWCbdu2YfLkyRg+fDjGjx+P1atXB7tJFKJ2796NBx98EKNHj8aIESNw2223Yfv27V51Dh8+jKlTp+LSSy9FdnY2Xn75ZciyHKQWU6ibPXs2+vXrhzNnznjKqqur8fDDD2PUqFEYMWIEHnroIZjN5iC2kkLNZ599httvvx0jR47E0KFD8dvf/tZre1FREWbOnIkRI0bgsssuw6JFi2C1WoPUWgpFX3zxBSZPnoxLL70UY8aMwcKFC1FaWurZzj5ETZWXl+O2227D3Xff7bOtNb+zZFnGP//5T1xxxRUYPnw47rzzThw6dKirmk8hoLk+VFVVhWeffRYTJkxAZmYmxo0bh9dee83nb2er1Yqnn37a8zf4jBkzUFRU1JWXQEF2rvehpnJyctC3b1+89tprPtvee+89XH311Rg+fDhuvfVWbNu2rbOaSwSAAV6HKygowKxZszBr1iz8+OOPePXVV/Gvf/0La9euDXbTKAQdP34cEyZMwJdffomtW7fi17/+NaZPn+75A6KyshJ33XUXrr/+enz//fd47733sGHDBr+/QIg+//xzAK4PNg6Hw1M+Z84caDQarF+/Hhs3boRGo8GDDz4YpFZSqFmyZAnefPNNzJs3D99++y22b9+OpUuXerbbbDbcfffduOSSS7BlyxZ8/vnnOH78OJ566qkgtppCyeeff45FixZh3rx5+P777/Hhhx+irKwMs2bNAsA+RN4KCgpw++23Q6lUwm63+2xvze+s1157DRs3bsTq1avx/fffY9KkSfj973+PysrKLroKCqZz9aGqqiro9Xq8/vrr2LZtG1555RWsXr0ab7/9tle9p556CidOnMBnn32GLVu2YMCAAbj77rths9m68lIo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"/>

<br/>

#### 5.3 Sex

<br/>

- 남자 = 1, 여자 = 0

- 남자보다 여자 생존자가 더 많다.

<br/>

```python
bar_chart('Sex')
```

<br/>

<img src="data:image/png;base64,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"/>

<br/>

#### 5.4 Age

<br/>

- 15 ~ 35 정도의 나이까지는 생존 확률보다 사망 확률이 높다.

- 어린아이, 연장자 위주로 탈출시켰을 것으로 추측된다.

<br/>

```python
snake = sns.FacetGrid(train,hue='Survived',aspect=4)
snake.map(sns.kdeplot,'Age',shade=True)
snake.set(xlim=(0,train['Age'].max()))
snake.add_legend()
plt.xlim(0,80)
plt.show()
```

<br/>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABPEAAAEcCAYAAACmvyFjAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACM9klEQVR4nOz9d3hc9Z3/fz/POdOlGfVuWe4d40oxxYBNCgHSSEgnuySBQLLszX43LD9IQraQXUg2YSFkd5OQZBOSkELooRgMBJtmGxewcZO7Lcuy6mj6Oef+YyTZQrIlW7Ik26/Hdc010plTPmPeSEev+RTDdV0XERERERERERERGbHM4W6AiIiIiIiIiIiIHJ1CPBERERERERERkRFOIZ6IiIiIiIiIiMgIpxBPRERERERERERkhFOIJyIiIiIiIiIiMsIpxBMRERERERERERnhFOKJiIiIiIiIiIiMcArxRERERERERERERjiFeCIiIiIiIiIiIiOcQjwREREREREREZERzjPcDei0YsUK7rrrLrZt20Z+fj5f+tKXuPrqq496zJIlS7jnnnuoq6ujvLycm266icWLF3e9XldXx0UXXURubm6342bPns1PfvKTE/I+REREREREREREBtuICPF27tzJjTfeyF133cXChQupra3luuuuIycnh8svv7zXY1auXMkdd9zB/fffz8yZM1m9ejU33HADBQUFzJ07F4BMJoNpmqxYsWIo346IiIiIiIiIiMigGhEh3q9//WuuvvpqFi5cCMC4ceO4/fbbueeee44Y4j3wwAN8/etfZ+bMmQDMmjWLG2+8kV/84hddId6J4roujY3tOI57Qq8jJyfTNCgszFGNyBGpRqQvqhHpD9WJ9EU1In1RjUhfVCPSF9M0KCrK7XtHGRQjYk68pUuXsmjRom7bFixYQG1tLfX19T32T6VSLFu2rMcxixcvZtmyZaTT6RPaXsMwME3jhF5DTl6maahG5KhUI9IX1Yj0h+pE+qIakb6oRqQvqhHpi2pjaA17Tzzbttm1axfjxo3rtt3r9TJq1Cg2b95MaWlpt9f279+P1+uluLi42/aysjJc12XPnj2MGTPmhLbbskZE/ikjUGdtqEbkSFQj0hfViPSH6kT6ohqRvqhGpC+qEemLamNoDXuI19zcDEA4HO7xWjgcpqWlpcf2pqamXvd/7zGGYeA4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rFjR7VFZWclPfvITnnvuuRP6nkREZHBF42l++Ps1+H0WHzl/rFbCOkG8HpP3zatm065mXlm7b7ibIyIiIiIiHUbEX0DXXnstDz30EC+99BIAtbW13HnnnVx33XVAdgXbL37xi9TW1nYdc/3113Pvvfeydu1aANasWcN9993HV77ylaF/AyIickJlbIcfPbyO1liaj184noBv2Kd0PaXVlIeZPqaAh17YQmt7aribIyIiIiIijICFLQAmTZrED3/4Q+6++25uvvlm8vLyuOaaa7jqqquA7OIXtbW1RKPRrmPOP/98br31Vm655Rbq6+spKSnhtttuY8GCBUe9ls/nw+fzndD3IyIig8d1XX759Lts2dPC1ZdMoCA8uKuNS+8unl1F7b53+d3zm/nKldOHuzkiIiIiIqc9w9WEN8elqaldq/NIrzwek4KCHNWIHJFq5Ng8+ep2/vRSLR86t4bpYwqHuzn949h42uuxUq2YyShmqg0rFcVMRTGTbVjpKK5h4XhDuJ4AjieI6w3gdH6dU4i/ahJtKWtYV95dV3uQv7y+k3+4ehbTx54k//anEf0skb6oRqQvqhHpi2pE+tJZIzI0RkRPPBERkd68+W49f3qplgUzykd0gGemoviat3c8tuFt2YXppLted00PjjeE4wnieIPY3hCG42Cm2jESTZh2CiOTwrCTGHYKAxfeBH+wkFReDelINam8atKRUbiewJC9rxljC3lneyO/fPpd/u3LZ+P1WEN2bRERERER6U4hnoiIjEibdjXzk8ffYVpNAefNKB/u5nRj2CkCB94hcGA9vubteGLZ1dJtb4hMbgWxqvlkcsqwfbk4niBY3v6f3HXxplsIpRtxGvfijdYRrF+H4WRwMUiHK4mXzyZeMRs7eGKDTcMwuHReNb/4y7v85bWdXHn+2BN6PREREREROTKFeCIiMuLsbWjnv/64lsriHD5w9mgMwxjuJoFj4z+4iVDdKgL712LaKTKhYtK5FcTKZ5POLcfxhWGgbTUMnEA+dmEZ8fA4bAdwHax4E972/XhbdhHZ+jR5m58gmT+GeMVc4mVn4vjDg/I236soEmDe5FKefHUH58wopzQ/eEKuIyIiIiIiR6cQT0RERpTmaJL//P1qcoIePnL+WDzWMC6k7rr4mrcT3LeKYN1bWOl2MoEC4uWzSBROxAnkD007DBM7VIQd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MWPGsGPHjh7nBRg7diybNm06rhDP4/Myff7sYz5OREREREREZHh0hFldg/HcrtDGwO0KaYyjLkMhIsNpRIR4zc3NAITD4R6vhcNhWlpaemxvamrqdf/3HtPY2HjE83Ze91iZpkUwr+C4jhURERERERERETlWI2IJlkwmg+u69DY935Gm7Eunj9wd2HXdrkklO899tH1ERERERERERERGshER4nX2lGtra+vxWltbG5FIpMf2SCRCa2trr+eLRqNd5wyHw8d0XhERERERERERkZFmRIR4oVCI0tJStm3b1m17Op1m9+7d1NTU9DimurqaWCxGQ0NDt+11dXWk02mqqqoAGDNmTI/zAmzbtq3X84qIiIiIiIiIiIw0IyLEg+xKtEuWLOm2bdmyZZSWllJdXd1j/0AgwJw5c3oc8/zzzzNv3jx8Pt8Rz7t582YaGhq0Mq2IiIiIiIiIiJwURkyId+211/LQQw/x0ksvAVBbW8udd97JddddB2RXsP3iF79IbW1t1zHXX3899957L2vXrgVgzZo13HfffXzlK1/p2uczn/kMy5cv589//jOu67J//35uu+02vvjFLxIIBIbwHYqIiIiIiIiIiBwfwz3SyhHDYPny5dx9993s3LmTvLw8rrnmGq655hoAkskkl156Kffddx8zZ87sOuaJJ57gRz/6EfX19ZSUlPC1r32Nyy+/vNt533nnHf7t3/6NTZs2EQgE+PjHP87f/d3fYVnWkL4/ERERERERERGR4zGiQjwRERERERERERHpacQMpxUREREREREREZHeKcQTEREREREREREZ4RTiiYiIiIiIiIiIjHAK8UREREREREREREY4hXgiIiIiIiIiIiIjnEI8ERERERERERGR/3979x4UZd3+cfyDigKKplNiDp5TUUdzDAWJ2hIzU/JQpgiTWEmmMOrIjOh4aFIxHw9j00xldhw1EQQPqGRqKGYqBqWUY2JqIB5WYVWgVE7374+mrX0Wf9VM7r2P+37N7B97fb/LXDvzmfs7XNzsujmGeAAAAAAAAICbY4j3D+Xl5Wns2LHq37+/nnrqKaWmpprdEkx07do1RUdHa9KkSU5rlZWVmj17tgYOHKiQkBDNmjVLFRUVJnQJsxw/flwzZsxQeHi4QkJCFB0drfz8fIc9P/30k2JjYzVgwABZLBa9++67MgzDpI7havv3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"/>

<br/>

#### 5.5 sibsp(친척) + parch(가족) = Together

<br/>

- 혼자 탄 경우가 사망 확률이 월등히 높다.

- 1~3인 동반자를 같이한 경우가 생존 확률이 높다.

<br/>

```python
train['Together']= train['SibSp'] + train['Parch']
train.drop('SibSp',axis=1,inplace=True)
train.drop('Parch',axis=1,inplace=True)

bar_chart('Together')
```

<br/>

<img src="data:image/png;base64,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tGbNmoZlXbt21SuvvKK+ffsqJiZG+/bt0+OPP67f/va3WrlypWJiYuTxeOTzNR0mwefzhXzLOdL+aNB2BXJBANA6Rdt5Hm3HGyivt20MmL9795cqLt6nnJxLmqy76abr1b//mVq4cEnI+wk1zAcdUouLi3XPPfdo3rx56ty5c8Nyu92uM888s+F179699dhjj2nkyJH6/PPPNWDAADmdTrndTYdocLvdcrlcwZYEAACAH3HddTfouutuaLJ8xIhB+vvfV6lz58gYDjSokFpVVaXx48drwoQJGjp06I9uHxcXp9NOO00lJSWSjj17+s477zTZrrCwUL179w6mJAAAAEt17OKMin22lIBDan19vSZOnKiMjAzdeOONfr2nqqpKhYWF6tu3ryQpIyNDc+fOVWVlZaMe/uvWrdPNN98caEkAAACWMU2fvF5T19w40JL9e71mSFOiHhcX104xMZEzGWnAlUyfPl0Oh0NTp0494fr9+/errKxM55xzjiRp586dmjVrlkaOHKl+/fpJknr27KmsrCxNmzZNc+fOlcPh0NKlS1VVVaUrrrgihMMBAABoWabpU0VFjQzDmiGqTNMXlpCal+f/mPctIaCQWlVVpddee00JCQkaPHhwo3VDhgzRk08+qcOHD+uhhx5SUVGRYmNj1bVrV/3yl7/U2LFjG23/yCOP6I9//KOys7MbpkVdtmyZ4uLiQj8qAACAFhSuoIjvBRRSnU6ndu7cecpt+vfvr1deeeVH20pISNBDDz2khx56KJASAAAAEAWsH30WAAAA+AFCKgAAACIOIRUAAAARh5AKAACAiENIBQAAQMQhpAIAACDiEFIBAAAQcSJn7isAAIBWyjBsrX7GqUhDSAUAAAiBYdiU0t4hw263ZP+m16vyitqAguqOHdt1112/VUJCQqPlWVmX6b77poW7xKAQUgEAAEJgGDYZdrveefFplR/4tkX3ndK5my4bM16GYQsopHq9HnXr1l0vvvhq8xUXIkIqAABAGJQf+Fbf/bvI6jLaDDpOAQAAIOIQUgEAABBx+LkfAAAgythsNpWVHdJNN12v7777TqmpqRo58kKNG3drk85UViGkAgAARJkzzzxbTz/9F/Xq1VuS9M03hVq4cL4effRBzZ07z+LqjuHnfgAAgCgTHx+vPn36KiYmRjExMTr99H6aOXOuNm4s0KFDB60u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"/>

<br/>

<br/>

## 6 데이터 전처리 함수화

<br/>

#### 6.1 위 전처리 작업들을 함수화

<br/>


- fillna : 결측값 대체 및 조정 함수

- drop_features : 불필요한 칼럼 제거함수

- format_features : 데이터 값 요약 및 인코딩 함수

- transform_features : 위 3개 함수를 모두 실행시키는 함수

- binning_features : 데이터 구간화 함수

<br/>

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

def binning_features(df):
  ranges=[0,15,25,35,60,100]
  labels=[0,1,2,3,4]
  df["AgeGroup"]=pd.cut(df['Age'],ranges,right=False,labels=labels)

  ranges=[0,15,30,100,1000]
  labels=[0,1,2,3]
  df["FareGroup"]=pd.cut(df['Fare'],ranges,right=False,labels=labels)
```

<br/>

#### 6.2 머신러닝을 위한 train / test 데이터 셋 분할 및 함수 적용

<br/>

```python
df_train = pd.read_csv('/content/train.csv')
df_test = pd.read_csv("/content/test.csv")

X_df = df_train.drop('Survived', axis=1)
X_df = transform_features(X_df)
target = df_train['Survived']

Z_df = transform_features(df_test)

binning_features(X_df)
binning_features(Z_df)
```

<pre>
데이터셋 null 개수: 0
데이터셋 null 개수: 0
</pre>
<br/>

#### 6.3 train / test 데이터 셋 칼럼 일치시키기

<br/>

```python
X_df = X_df[["PassengerId", "Pclass", "Sex", "AgeGroup", "FareGroup"]]
Z_df = Z_df[["PassengerId", "Pclass", "Sex", "AgeGroup", "FareGroup"]]
```


```python
X_df.head()
```

<br/>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>AgeGroup</th>
      <th>FareGroup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<br/>

```python
Z_df.head()
```

<br/>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>AgeGroup</th>
      <th>FareGroup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<br/>

<br/>

## 7. 머신러닝

<br/>


- 고려해야 할 변수 : Pclass, FaerGroup, AgeGroup, Sex

  <br/>

#### 7.1 Test set / Train set 분리

<br/>

- 비율은 8:2로 설정한다.

- X_train / X_test : 생존 칼럼을 제외한 나머지 데이터 8:2

- y_train / y_test : 생존 칼럼만 추출한 데이터 8:2

<br/>

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_df, target, test_size=0.2, random_state=2024)
print(len(X_train), len(y_train))
print(len(X_test), len(y_test))
```

<pre>
712 712
179 179
</pre>
<br/>

#### 7.2 모델 정확도 측정

<br/>

- 단순하게 각 모델 별로 정확도를 측정해본다.

- dt_clf : 의사결정나무

- rf_clf : 랜덤 포레스트

- lr_clf : 로지스틱 회귀분석

<br/>

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 결정트리, Random Forest, 로지스틱 회귀를 위한 사이킷런 Classifier 클래스 생성
dt = DecisionTreeClassifier(random_state=2024) #결정트리
rf = RandomForestClassifier(random_state=2024) #Random Forest
lr = LogisticRegression(solver='liblinear') #로지스틱 회귀

dt.fit(X_train, y_train) # train 데이터 학습 피팅
dt_pred = dt.predict(X_test) # 테스트 데이터로 확인
print('DT 정확도: {0:.4f}'.format(accuracy_score(y_test, dt_pred))) # 정확도 예측값 출력

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print('RF 정확도: {0:.4f}'.format(accuracy_score(y_test, rf_pred)))

lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print('LR 정확도: {0:.4f}'.format(accuracy_score(y_test, lr_pred)))
```

<pre>
DT 정확도: 0.7542
RF 정확도: 0.8156
LR 정확도: 0.7877
</pre>
<br/>

- RF(랜덤 포레스트) 정확도 81%로 가장 우수한 정확도 나타냈다.

- 하지만 아직 최적화 작업을 수행하지 않았고, 데이터 양도 충분치 않기 때문에 어떤 알고리즘이 가장 좋은 성능인지 평가할 수는 없다.

<br/>

#### 7.3 교차검증 ( K-Fold )

<br/>

- 데이터 편중을 막기 위해 별도의 세트로 구성된 학습 데이터 세트와 검증 데이터 세트에서 평가 수행

- k-fold 갯수 : 5

<br/>


```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split

def exec_kfold(clf, folds=5):
  kfold = KFold(n_splits=folds, shuffle = True, random_state = 42)
  scores = []

  for i, (train_index, test_index) in enumerate(kfold.split(X_df)): # K-Fold 반복
    X_train, X_test = X_df.values[train_index], X_df.values[test_index] # 8의 비율을 가진 데이터
    y_train, y_test = target.values[train_index], target.values[test_index] # 2의 비율을 가진 데이터

    clf.fit(X_train, y_train) # Classifier 학습, 예측, 정확도 계산
    prediction = clf.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    scores.append(accuracy)
    print('교차 검증{0} 정확도:{1:4f}'.format(i, accuracy))

  mean_score = np.mean(scores) # 5개 fold에서 평균 정확도 계산
  print('평균정확도:{0:4f}'.format(mean_score))

# 함수 호출
print("[dt : 의사결정나무 정확도]")
print(exec_kfold(dt, folds=5))
print("-"*50)
print("[rf : 랜덤 포레스트 정확도]")
print(exec_kfold(rf, folds=5))
print("-"*50)
print("[rl : 로지스틱 회귀분석 정확도]")
print(exec_kfold(lr, folds=5))
print("-"*50)
```

<pre>
[dt : 의사결정나무 정확도]
교차 검증0 정확도:0.715084
교차 검증1 정확도:0.713483
교차 검증2 정확도:0.752809
교차 검증3 정확도:0.724719
교차 검증4 정확도:0.775281
평균정확도:0.736275
None
--------------------------------------------------
[rf : 랜덤 포레스트 정확도]
교차 검증0 정확도:0.793296
교차 검증1 정확도:0.758427
교차 검증2 정확도:0.786517
교차 검증3 정확도:0.752809
교차 검증4 정확도:0.786517
평균정확도:0.775513
None
--------------------------------------------------
[rl : 로지스틱 회귀분석 정확도]
교차 검증0 정확도:0.787709
교차 검증1 정확도:0.769663
교차 검증2 정확도:0.848315
교차 검증3 정확도:0.730337
교차 검증4 정확도:0.803371
평균정확도:0.787879
None
--------------------------------------------------
</pre>
<br/>

- 평균 정확도에서는 78%로 로지스틱 회귀분석이 높게 책정되나

- 초기 측정값의 차이 대비 랜덤 포레스트에서도 유사한 관측값을 보여 랜덤 포레스트로 검증 진행

<br/>

#### 7.4 Cross_val_score 교차검증

<br/>

- 불균형한 분포도를 가진 레이블 데이터 집합을 위한 KFold 방식

- 특정 레이블 값이 특이하게 많거나, 매우 적어서 값의 분포가 한쪽으로 치우치는 경우에 적합

- kFold와 cross_val_score의 점수가 다른 것은, cross_val_score 는 StratifiedKFold를 이용해 세트를 분할하기 때문

<br/>

```python
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=5, random_state=2024, shuffle=True) # shuffle 랜덤하게 데이터셋을 섞음
accs = cross_val_score(rf, X_df, target, cv=cv) # model, X, y, cv
print(accs)
print('평균정확도:', np.mean(accs))
```

<pre>
[0.8156 0.764  0.8034 0.7247 0.8034]
평균정확도: 0.7822296152156174
</pre>
<br/>

**[Part 7. 결론]**

- 모델 : 랜덤 포레스트

- K-Fold 평균 정확도 : 0.7755

- Cross_val_score 평균 정확도 : 0.7822

  <br/>

  <br/>


## 8. 최적 파라미터

<br/>

#### 8.1 최적 파라미터 탐색

<br/>


- 그리드 서치(Grid Search)를 사용하여 랜덤 포레스트 모델의 최적 하이퍼파라미터를 찾는다.

- 그리고 그에 대한 모델을 훈련하고 테스트 세트에서의 정확도를 출력하는 작업을 진행한다.

<br/>

```python
from sklearn.model_selection import GridSearchCV # 최적 하이퍼파라미터를 찾는 알고리즘

parameters = {'max_depth':[2,3,5,10], # 높을수록 모델이 복잡해지고 과적합 우려 존재
              'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]} # 높이면 덜 분할 일반화, 높이면 더 작은 리프노드, 간단해짐

grid_dclf = GridSearchCV(rf, param_grid=parameters, scoring='accuracy', cv=5) # n_jobs=-1(모든 cpu 사용), verbose=2(로그출력)
grid_dclf.fit(X_train, y_train)

print('최적 하이퍼 파라미터:', grid_dclf.best_params_)
print('최고 정확도:{0:4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_

dpred = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test, dpred)
print('테스트셋 데이터의 예측 수행한 정확도:{0: 4f}'.format(accuracy))
```

<pre>
최적 하이퍼 파라미터: {'max_depth': 3, 'min_samples_leaf': 5, 'min_samples_split': 2}
최고 정확도:0.800552
테스트셋 데이터의 예측 수행한 정확도: 0.826816
</pre>
<br/>

#### 8.2 최적 파라미터 적용

<br/>

```python
pred = best_dclf.predict(Z_df)

pred
```

<pre>
array([0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0,
       1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,
       1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
       1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0,
       1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
       1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
       0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
       0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0,
       0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
       1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
       0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0])
</pre>
<br/>

<br/>

## 9. 제출 파일 형식

<br/>

-  PassengerId(승객 ID)와 Survived(생존 여부)  두 개의 column으로 이루어진 csv 파일 제출

  - 418개의 항목과 헤더 행이 포함된 csv 파일을 제출

  - test data 내에 PassengerId(승객 ID)

  - test data 에 modeling을 적용하여 얻은‘Survived(생존 여부) 값

<br/>

```python
submission = pd.DataFrame({
"PassengerId":test['PassengerId'],
"Survived": pred
})

submission.to_csv('submission.csv',index=False)
```


```python
submissionfile = pd.read_csv('/content/submission.csv')

print(submissionfile.head())
```

<pre>
   PassengerId  Survived
0          892         0
1          893         0
2          894         0
3          895         0
4          896         1
</pre>