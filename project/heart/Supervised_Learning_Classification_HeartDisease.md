# Supervised Learning - Classification of Heart Disease
1. Import libraries, set up directory, and read data
2. Quick data check 
3. Modeling
    - Logistic regression: StandardScaler, LogisticRegressionCV
    - KNN: Pipe(StandardScaler, KNeighborsClassifier), GridsearchCV
    - SVM
    - Decision tree
    - Random forest
    - Adaboost
    - Gradient boosting
    - XGBoost
4. Compare the results
5. Save and load a final model

## 1. Import libraries, set up directories, and read data


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
mypath = '/content/drive/MyDrive/Udemy/2022_PythonForMLDSMasterClass/project/'
```


```python
df = pd.read_csv(mypath + 'heart.csv')
```

## 2. Quick data check

The data used here already organized through feature engineering and data cleaning. 

Original Source at https://archive.ics.uci.edu/ml/datasets/Heart+Disease




```python
print(df.info())
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 303 entries, 0 to 302
    Data columns (total 14 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       303 non-null    int64  
     1   sex       303 non-null    int64  
     2   cp        303 non-null    int64  
     3   trestbps  303 non-null    int64  
     4   chol      303 non-null    int64  
     5   fbs       303 non-null    int64  
     6   restecg   303 non-null    int64  
     7   thalach   303 non-null    int64  
     8   exang     303 non-null    int64  
     9   oldpeak   303 non-null    float64
     10  slope     303 non-null    int64  
     11  ca        303 non-null    int64  
     12  thal      303 non-null    int64  
     13  target    303 non-null    int64  
    dtypes: float64(1), int64(13)
    memory usage: 33.3 KB
    None






  <div id="df-4b89951e-249f-46b3-b69e-d73141af7be7">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4b89951e-249f-46b3-b69e-d73141af7be7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-4b89951e-249f-46b3-b69e-d73141af7be7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4b89951e-249f-46b3-b69e-d73141af7be7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## 3. Modeling


```python
X = df.drop('target', axis=1)
y = df['target']
```


```python
X.head()
```





  <div id="df-68aaa9b8-d989-476d-b1ae-d8566d2d9b9c">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-68aaa9b8-d989-476d-b1ae-d8566d2d9b9c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-68aaa9b8-d989-476d-b1ae-d8566d2d9b9c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-68aaa9b8-d989-476d-b1ae-d8566d2d9b9c');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```


```python
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score

def display_evaluation(y_test, y_pred):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    print(classification_report(y_test, y_pred))

res = {}
def save_results(model_name, y_test, y_pred):
    res[model_name] = accuracy_score(y_test, y_pred)
```

### 3.1. Logistic regression


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
```


```python
from sklearn.linear_model import LogisticRegressionCV

grid_model = LogisticRegressionCV(Cs=10, penalty='elasticnet', solver='saga', l1_ratios=np.linspace(0, 1, 11, endpoint=True), max_iter=5000)
grid_model.fit(scaled_X_train, y_train)
```




    LogisticRegressionCV(l1_ratios=array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                         max_iter=5000, penalty='elasticnet', solver='saga')




```python
grid_model.C_
```




    array([0.35938137])




```python
grid_model.l1_ratio_ # l2 regularization
```




    array([0.1])




```python
grid_model.coef_
```




    array([[-0.18412756, -0.64175712,  0.79616508, -0.28674134, -0.19690496,
             0.18239095,  0.18426002,  0.26737504, -0.4152586 , -0.53320765,
             0.33957311, -0.63575143, -0.46679496]])




```python
coefs = pd.Series(data=grid_model.coef_[0], index=X.columns).sort_values()
```


```python
coefs
```




    sex        -0.641757
    ca         -0.635751
    oldpeak    -0.533208
    thal       -0.466795
    exang      -0.415259
    trestbps   -0.286741
    chol       -0.196905
    age        -0.184128
    fbs         0.182391
    restecg     0.184260
    thalach     0.267375
    slope       0.339573
    cp          0.796165
    dtype: float64




```python
sns.barplot(x=coefs.index, y=coefs)
plt.ylabel('Coefficient')
plt.xticks(rotation=90);
```


    
![png](output_20_0.png)
    



```python
y_pred = grid_model.predict(scaled_X_test)
display_evaluation(y_test, y_pred)
save_results('Logistic Regression', y_test, y_pred)
```

                  precision    recall  f1-score   support
    
               0       0.89      0.77      0.83        44
               1       0.81      0.91      0.86        47
    
        accuracy                           0.85        91
       macro avg       0.85      0.84      0.84        91
    weighted avg       0.85      0.85      0.85        91
    



    
![png](output_21_1.png)
    


### 3.2 KNN



```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

from sklearn.pipeline import Pipeline
operations = [('scaler', scaler), ('model', model)]
pipe = Pipeline(operations)
```


```python
model.get_params()
```




    {'algorithm': 'auto',
     'leaf_size': 30,
     'metric': 'minkowski',
     'metric_params': None,
     'n_jobs': None,
     'n_neighbors': 5,
     'p': 2,
     'weights': 'uniform'}




```python
from sklearn.model_selection import GridSearchCV

param_grid = {'model__n_neighbors':np.arange(1, 30)}
grid_model = GridSearchCV(pipe, param_grid, scoring='accuracy')
grid_model.fit(X_train, y_train)
```




    GridSearchCV(estimator=Pipeline(steps=[('scaler', StandardScaler()),
                                           ('model', KNeighborsClassifier())]),
                 param_grid={'model__n_neighbors': array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])},
                 scoring='accuracy')




```python
grid_model.cv_results_.keys()
```




    dict_keys(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'param_model__n_neighbors', 'params', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score'])




```python
grid_model.cv_results_['mean_test_score']
```




    array([0.75470653, 0.75492802, 0.80664452, 0.80199336, 0.81616833,
           0.82093023, 0.81140642, 0.82070875, 0.8255814 , 0.84440753,
           0.80664452, 0.79723145, 0.78781838, 0.80177187, 0.79712071,
           0.81140642, 0.80653378, 0.82547065, 0.80199336, 0.81605759,
           0.81140642, 0.80653378, 0.80199336, 0.80653378, 0.7923588 ,
           0.80188261, 0.80188261, 0.79246955, 0.79246955])




```python
grid_model.best_params_
```




    {'model__n_neighbors': 10}




```python
plt.plot(np.arange(1, 30), grid_model.cv_results_['mean_test_score'], 'o-')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('Using Grid Search');
```


    
![png](output_29_0.png)
    



```python
y_pred = grid_model.predict(X_test)
display_evaluation(y_test, y_pred)
```

                  precision    recall  f1-score   support
    
               0       0.82      0.84      0.83        44
               1       0.85      0.83      0.84        47
    
        accuracy                           0.84        91
       macro avg       0.84      0.84      0.84        91
    weighted avg       0.84      0.84      0.84        91
    



    
![png](output_30_1.png)
    



```python
# Verify with Elbow method

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

from sklearn.metrics import accuracy_score
accs = [] 
for k in range(1, 31):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(scaled_X_train, y_train)
    y_pred = model.predict(scaled_X_test)
    acc = accuracy_score(y_test, y_pred)
    accs.append(acc)
```


```python
plt.plot(range(1, 31), accs, 'o-')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('Using Manual Elbow Method');
```


    
![png](output_32_0.png)
    



```python
model = KNeighborsClassifier(n_neighbors=7)
model.fit(scaled_X_train, y_train)
```




    KNeighborsClassifier(n_neighbors=7)




```python
y_pred = model.predict(scaled_X_test)
display_evaluation(y_test, y_pred)
save_results('KNN', y_test, y_pred)
```

                  precision    recall  f1-score   support
    
               0       0.90      0.84      0.87        44
               1       0.86      0.91      0.89        47
    
        accuracy                           0.88        91
       macro avg       0.88      0.88      0.88        91
    weighted avg       0.88      0.88      0.88        91
    



    
![png](output_34_1.png)
    


### 3.3 SVM


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.svm import SVC
model = SVC()

from sklearn.pipeline import Pipeline
operations = [('scaler', scaler), ('model', model)]
pipe = Pipeline(operations)
```


```python
from sklearn.model_selection import GridSearchCV

C_values = [0.001, 0.01, 0.1, 0.5, 1]
gamma_values = ['scale', 'auto']
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = {'model__C':C_values, 'model__kernel': kernels, 'model__gamma': gamma_values}
grid_model = GridSearchCV(pipe, param_grid, scoring='accuracy')
grid_model.fit(X_train, y_train)
```




    GridSearchCV(estimator=Pipeline(steps=[('scaler', StandardScaler()),
                                           ('model', SVC())]),
                 param_grid={'model__C': [0.001, 0.01, 0.1, 0.5, 1],
                             'model__gamma': ['scale', 'auto'],
                             'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                 scoring='accuracy')




```python
grid_model.cv_results_.keys()
```




    dict_keys(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'param_model__C', 'param_model__gamma', 'param_model__kernel', 'params', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score'])




```python
grid_model.cv_results_['mean_test_score']
```




    array([0.55658915, 0.55658915, 0.55658915, 0.55658915, 0.55658915,
           0.55658915, 0.55658915, 0.55658915, 0.81140642, 0.55658915,
           0.55658915, 0.55658915, 0.81140642, 0.55658915, 0.55658915,
           0.55658915, 0.83045404, 0.66943522, 0.75481728, 0.81140642,
           0.83045404, 0.66943522, 0.75481728, 0.81140642, 0.82104097,
           0.73554817, 0.79723145, 0.80199336, 0.82104097, 0.73554817,
           0.79723145, 0.80199336, 0.82115172, 0.76866002, 0.78327796,
           0.80675526, 0.82115172, 0.76866002, 0.78327796, 0.80675526])




```python
grid_model.best_params_
```




    {'model__C': 0.1, 'model__gamma': 'scale', 'model__kernel': 'linear'}




```python
plt.plot(range(1, 41), grid_model.cv_results_['mean_test_score'], 'o-')
plt.ylabel('Mean Test Score');
```


    
![png](output_41_0.png)
    



```python
grid_model.cv_results_['params']
```




    [{'model__C': 0.001, 'model__gamma': 'scale', 'model__kernel': 'linear'},
     {'model__C': 0.001, 'model__gamma': 'scale', 'model__kernel': 'poly'},
     {'model__C': 0.001, 'model__gamma': 'scale', 'model__kernel': 'rbf'},
     {'model__C': 0.001, 'model__gamma': 'scale', 'model__kernel': 'sigmoid'},
     {'model__C': 0.001, 'model__gamma': 'auto', 'model__kernel': 'linear'},
     {'model__C': 0.001, 'model__gamma': 'auto', 'model__kernel': 'poly'},
     {'model__C': 0.001, 'model__gamma': 'auto', 'model__kernel': 'rbf'},
     {'model__C': 0.001, 'model__gamma': 'auto', 'model__kernel': 'sigmoid'},
     {'model__C': 0.01, 'model__gamma': 'scale', 'model__kernel': 'linear'},
     {'model__C': 0.01, 'model__gamma': 'scale', 'model__kernel': 'poly'},
     {'model__C': 0.01, 'model__gamma': 'scale', 'model__kernel': 'rbf'},
     {'model__C': 0.01, 'model__gamma': 'scale', 'model__kernel': 'sigmoid'},
     {'model__C': 0.01, 'model__gamma': 'auto', 'model__kernel': 'linear'},
     {'model__C': 0.01, 'model__gamma': 'auto', 'model__kernel': 'poly'},
     {'model__C': 0.01, 'model__gamma': 'auto', 'model__kernel': 'rbf'},
     {'model__C': 0.01, 'model__gamma': 'auto', 'model__kernel': 'sigmoid'},
     {'model__C': 0.1, 'model__gamma': 'scale', 'model__kernel': 'linear'},
     {'model__C': 0.1, 'model__gamma': 'scale', 'model__kernel': 'poly'},
     {'model__C': 0.1, 'model__gamma': 'scale', 'model__kernel': 'rbf'},
     {'model__C': 0.1, 'model__gamma': 'scale', 'model__kernel': 'sigmoid'},
     {'model__C': 0.1, 'model__gamma': 'auto', 'model__kernel': 'linear'},
     {'model__C': 0.1, 'model__gamma': 'auto', 'model__kernel': 'poly'},
     {'model__C': 0.1, 'model__gamma': 'auto', 'model__kernel': 'rbf'},
     {'model__C': 0.1, 'model__gamma': 'auto', 'model__kernel': 'sigmoid'},
     {'model__C': 0.5, 'model__gamma': 'scale', 'model__kernel': 'linear'},
     {'model__C': 0.5, 'model__gamma': 'scale', 'model__kernel': 'poly'},
     {'model__C': 0.5, 'model__gamma': 'scale', 'model__kernel': 'rbf'},
     {'model__C': 0.5, 'model__gamma': 'scale', 'model__kernel': 'sigmoid'},
     {'model__C': 0.5, 'model__gamma': 'auto', 'model__kernel': 'linear'},
     {'model__C': 0.5, 'model__gamma': 'auto', 'model__kernel': 'poly'},
     {'model__C': 0.5, 'model__gamma': 'auto', 'model__kernel': 'rbf'},
     {'model__C': 0.5, 'model__gamma': 'auto', 'model__kernel': 'sigmoid'},
     {'model__C': 1, 'model__gamma': 'scale', 'model__kernel': 'linear'},
     {'model__C': 1, 'model__gamma': 'scale', 'model__kernel': 'poly'},
     {'model__C': 1, 'model__gamma': 'scale', 'model__kernel': 'rbf'},
     {'model__C': 1, 'model__gamma': 'scale', 'model__kernel': 'sigmoid'},
     {'model__C': 1, 'model__gamma': 'auto', 'model__kernel': 'linear'},
     {'model__C': 1, 'model__gamma': 'auto', 'model__kernel': 'poly'},
     {'model__C': 1, 'model__gamma': 'auto', 'model__kernel': 'rbf'},
     {'model__C': 1, 'model__gamma': 'auto', 'model__kernel': 'sigmoid'}]




```python
scaler = StandardScaler()
model = SVC(C=0.1, kernel='linear', gamma='scale')
operations = [('scaler', scaler), ('model', model)]
pipe = Pipeline(operations)
pipe.fit(X_train, y_train)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('model', SVC(C=0.1, kernel='linear'))])




```python
y_pred = pipe.predict(X_test)
display_evaluation(y_test, y_pred)
save_results('SVM', y_test, y_pred)
```

                  precision    recall  f1-score   support
    
               0       0.92      0.77      0.84        44
               1       0.81      0.94      0.87        47
    
        accuracy                           0.86        91
       macro avg       0.87      0.85      0.86        91
    weighted avg       0.87      0.86      0.86        91
    



    
![png](output_44_1.png)
    


### 3.4 Decision tree



```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

from sklearn.model_selection import GridSearchCV

param_grid = {'criterion':['gini', 'entropy'], 'max_depth':[2,3,4,5], 'max_leaf_nodes':np.arange(5, 15)}
grid_model = GridSearchCV(model, param_grid, scoring='accuracy')

grid_model.fit(X_train, y_train)
```




    GridSearchCV(estimator=DecisionTreeClassifier(),
                 param_grid={'criterion': ['gini', 'entropy'],
                             'max_depth': [2, 3, 4, 5],
                             'max_leaf_nodes': array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14])},
                 scoring='accuracy')




```python
grid_model.best_params_
```




    {'criterion': 'gini', 'max_depth': 4, 'max_leaf_nodes': 11}




```python
plt.figure(figsize=(12, 5))
plt.plot(grid_model.cv_results_['mean_test_score'], '-o')
plt.ylabel('Mean Test Score');
```


    
![png](output_48_0.png)
    



```python
idxmax = grid_model.cv_results_['mean_test_score'].argmax()
```


```python
grid_model.cv_results_['params'][idxmax]
```




    {'criterion': 'gini', 'max_depth': 4, 'max_leaf_nodes': 11}




```python
model = DecisionTreeClassifier(criterion='gini', max_depth=4, max_leaf_nodes=11)
model.fit(X_train, y_train)
```




    DecisionTreeClassifier(max_depth=4, max_leaf_nodes=11)




```python
y_pred = model.predict(X_test)
display_evaluation(y_test, y_pred)
save_results('Decision Tree', y_test, y_pred)
```

                  precision    recall  f1-score   support
    
               0       0.81      0.77      0.79        44
               1       0.80      0.83      0.81        47
    
        accuracy                           0.80        91
       macro avg       0.80      0.80      0.80        91
    weighted avg       0.80      0.80      0.80        91
    



    
![png](output_52_1.png)
    



```python
from sklearn.tree import plot_tree

plt.figure(figsize=(10, 10))
plot_tree(model);
```


    
![png](output_53_0.png)
    


### 3.5 Random forest


```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
```


```python
model.get_params()
```




    {'bootstrap': True,
     'ccp_alpha': 0.0,
     'class_weight': None,
     'criterion': 'gini',
     'max_depth': None,
     'max_features': 'auto',
     'max_leaf_nodes': None,
     'max_samples': None,
     'min_impurity_decrease': 0.0,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 100,
     'n_jobs': None,
     'oob_score': False,
     'random_state': None,
     'verbose': 0,
     'warm_start': False}




```python
from sklearn.model_selection import GridSearchCV

#bootstrap = [True]
oob_score = [True, False]
n_estimators = [50, 64, 100, 128, 150, 200]
max_features = ['auto', 'log2'] # auto = sqrt(n)

param_grid = {'n_estimators':n_estimators, 'oob_score':oob_score, 'max_features': max_features}
grid_model = GridSearchCV(model, param_grid, scoring='accuracy')
grid_model.fit(X_train, y_train)
```




    GridSearchCV(estimator=RandomForestClassifier(),
                 param_grid={'max_features': ['auto', 'log2'],
                             'n_estimators': [50, 64, 100, 128, 150, 200],
                             'oob_score': [True, False]},
                 scoring='accuracy')




```python
grid_model.best_params_
```




    {'max_features': 'log2', 'n_estimators': 150, 'oob_score': False}




```python
grid_model.cv_results_['mean_test_score']
```




    array([0.7923588 , 0.8021041 , 0.81129568, 0.76921373, 0.79258029,
           0.82547065, 0.79734219, 0.80188261, 0.79712071, 0.81616833,
           0.80675526, 0.82081949, 0.81129568, 0.81594684, 0.81118494,
           0.81129568, 0.79723145, 0.79258029, 0.81594684, 0.79712071,
           0.80188261, 0.8303433 , 0.81140642, 0.80177187])




```python
plt.plot(grid_model.cv_results_['mean_test_score'], 'o-');
```


    
![png](output_60_0.png)
    



```python
y_pred = grid_model.predict(X_test)
display_evaluation(y_test, y_pred)
save_results('Random Forest', y_test, y_pred)
```

                  precision    recall  f1-score   support
    
               0       0.88      0.82      0.85        44
               1       0.84      0.89      0.87        47
    
        accuracy                           0.86        91
       macro avg       0.86      0.86      0.86        91
    weighted avg       0.86      0.86      0.86        91
    



    
![png](output_61_1.png)
    


### 3.6 Adaboost


```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

error_rates = []
ns = [10, 20, 50, 64, 100, 128, 150, 200, 256]
for n in ns:

    model = AdaBoostClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    err = 1 - accuracy_score(y_test, y_pred)
    error_rates.append(err)

```


```python
error_rates
```




    [0.18681318681318682,
     0.13186813186813184,
     0.19780219780219777,
     0.2417582417582418,
     0.2417582417582418,
     0.2417582417582418,
     0.2417582417582418,
     0.2417582417582418,
     0.2637362637362637]




```python
plt.figure(figsize=(10, 4))
plt.plot(ns, error_rates, 'o-')
plt.xticks(ns)
plt.xlabel('n_estimators')
plt.ylabel('Error Rates');
```


    
![png](output_65_0.png)
    



```python
model = AdaBoostClassifier(n_estimators=20)
model.fit(X_train, y_train)
```




    AdaBoostClassifier(n_estimators=20)




```python
y_pred = model.predict(X_test)
display_evaluation(y_test, y_pred)
save_results('Adaboost', y_test, y_pred)
```

                  precision    recall  f1-score   support
    
               0       0.88      0.84      0.86        44
               1       0.86      0.89      0.88        47
    
        accuracy                           0.87        91
       macro avg       0.87      0.87      0.87        91
    weighted avg       0.87      0.87      0.87        91
    



    
![png](output_67_1.png)
    


### 3.7 Gradient Boosting


```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()

from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [64, 100, 128], 'max_depth':np.arange(2, 11), 'max_leaf_nodes':np.arange(2, 21)} # max_depth, max_leaf_nodes
grid_model = GridSearchCV(model, param_grid, scoring='accuracy')
grid_model.fit(X_train, y_train)
```




    GridSearchCV(estimator=GradientBoostingClassifier(),
                 param_grid={'max_depth': array([ 2,  3,  4,  5,  6,  7,  8,  9, 10]),
                             'max_leaf_nodes': array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
           19, 20]),
                             'n_estimators': [64, 100, 128]},
                 scoring='accuracy')




```python
grid_model.best_params_
```




    {'max_depth': 2, 'max_leaf_nodes': 2, 'n_estimators': 100}




```python
grid_model.cv_results_['params']
```




    [{'max_depth': 2, 'max_leaf_nodes': 2, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 2, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 2, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 3, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 3, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 3, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 4, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 4, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 4, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 5, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 5, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 5, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 6, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 6, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 6, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 7, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 7, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 7, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 8, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 8, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 8, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 9, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 9, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 9, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 10, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 10, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 10, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 11, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 11, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 11, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 12, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 12, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 12, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 13, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 13, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 13, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 14, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 14, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 14, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 15, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 15, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 15, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 16, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 16, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 16, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 17, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 17, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 17, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 18, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 18, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 18, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 19, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 19, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 19, 'n_estimators': 128},
     {'max_depth': 2, 'max_leaf_nodes': 20, 'n_estimators': 64},
     {'max_depth': 2, 'max_leaf_nodes': 20, 'n_estimators': 100},
     {'max_depth': 2, 'max_leaf_nodes': 20, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 2, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 2, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 2, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 3, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 3, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 3, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 4, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 4, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 4, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 5, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 5, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 5, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 6, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 6, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 6, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 7, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 7, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 7, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 8, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 8, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 8, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 9, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 9, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 9, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 10, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 10, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 10, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 11, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 11, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 11, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 12, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 12, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 12, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 13, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 13, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 13, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 14, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 14, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 14, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 15, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 15, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 15, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 16, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 16, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 16, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 17, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 17, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 17, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 18, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 18, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 18, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 19, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 19, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 19, 'n_estimators': 128},
     {'max_depth': 3, 'max_leaf_nodes': 20, 'n_estimators': 64},
     {'max_depth': 3, 'max_leaf_nodes': 20, 'n_estimators': 100},
     {'max_depth': 3, 'max_leaf_nodes': 20, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 2, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 2, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 2, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 3, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 3, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 3, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 4, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 4, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 4, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 5, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 5, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 5, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 6, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 6, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 6, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 7, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 7, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 7, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 8, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 8, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 8, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 9, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 9, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 9, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 10, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 10, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 10, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 11, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 11, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 11, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 12, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 12, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 12, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 13, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 13, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 13, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 14, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 14, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 14, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 15, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 15, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 15, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 16, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 16, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 16, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 17, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 17, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 17, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 18, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 18, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 18, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 19, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 19, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 19, 'n_estimators': 128},
     {'max_depth': 4, 'max_leaf_nodes': 20, 'n_estimators': 64},
     {'max_depth': 4, 'max_leaf_nodes': 20, 'n_estimators': 100},
     {'max_depth': 4, 'max_leaf_nodes': 20, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 2, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 2, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 2, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 3, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 3, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 3, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 4, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 4, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 4, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 5, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 5, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 5, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 6, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 6, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 6, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 7, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 7, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 7, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 8, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 8, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 8, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 9, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 9, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 9, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 10, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 10, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 10, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 11, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 11, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 11, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 12, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 12, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 12, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 13, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 13, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 13, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 14, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 14, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 14, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 15, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 15, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 15, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 16, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 16, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 16, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 17, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 17, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 17, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 18, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 18, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 18, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 19, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 19, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 19, 'n_estimators': 128},
     {'max_depth': 5, 'max_leaf_nodes': 20, 'n_estimators': 64},
     {'max_depth': 5, 'max_leaf_nodes': 20, 'n_estimators': 100},
     {'max_depth': 5, 'max_leaf_nodes': 20, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 2, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 2, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 2, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 3, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 3, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 3, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 4, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 4, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 4, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 5, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 5, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 5, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 6, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 6, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 6, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 7, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 7, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 7, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 8, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 8, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 8, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 9, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 9, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 9, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 10, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 10, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 10, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 11, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 11, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 11, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 12, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 12, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 12, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 13, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 13, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 13, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 14, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 14, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 14, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 15, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 15, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 15, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 16, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 16, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 16, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 17, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 17, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 17, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 18, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 18, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 18, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 19, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 19, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 19, 'n_estimators': 128},
     {'max_depth': 6, 'max_leaf_nodes': 20, 'n_estimators': 64},
     {'max_depth': 6, 'max_leaf_nodes': 20, 'n_estimators': 100},
     {'max_depth': 6, 'max_leaf_nodes': 20, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 2, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 2, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 2, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 3, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 3, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 3, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 4, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 4, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 4, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 5, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 5, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 5, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 6, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 6, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 6, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 7, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 7, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 7, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 8, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 8, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 8, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 9, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 9, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 9, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 10, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 10, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 10, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 11, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 11, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 11, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 12, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 12, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 12, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 13, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 13, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 13, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 14, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 14, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 14, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 15, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 15, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 15, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 16, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 16, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 16, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 17, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 17, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 17, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 18, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 18, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 18, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 19, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 19, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 19, 'n_estimators': 128},
     {'max_depth': 7, 'max_leaf_nodes': 20, 'n_estimators': 64},
     {'max_depth': 7, 'max_leaf_nodes': 20, 'n_estimators': 100},
     {'max_depth': 7, 'max_leaf_nodes': 20, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 2, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 2, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 2, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 3, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 3, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 3, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 4, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 4, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 4, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 5, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 5, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 5, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 6, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 6, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 6, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 7, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 7, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 7, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 8, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 8, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 8, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 9, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 9, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 9, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 10, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 10, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 10, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 11, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 11, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 11, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 12, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 12, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 12, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 13, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 13, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 13, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 14, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 14, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 14, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 15, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 15, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 15, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 16, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 16, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 16, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 17, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 17, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 17, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 18, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 18, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 18, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 19, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 19, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 19, 'n_estimators': 128},
     {'max_depth': 8, 'max_leaf_nodes': 20, 'n_estimators': 64},
     {'max_depth': 8, 'max_leaf_nodes': 20, 'n_estimators': 100},
     {'max_depth': 8, 'max_leaf_nodes': 20, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 2, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 2, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 2, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 3, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 3, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 3, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 4, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 4, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 4, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 5, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 5, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 5, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 6, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 6, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 6, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 7, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 7, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 7, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 8, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 8, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 8, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 9, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 9, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 9, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 10, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 10, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 10, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 11, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 11, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 11, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 12, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 12, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 12, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 13, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 13, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 13, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 14, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 14, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 14, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 15, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 15, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 15, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 16, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 16, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 16, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 17, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 17, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 17, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 18, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 18, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 18, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 19, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 19, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 19, 'n_estimators': 128},
     {'max_depth': 9, 'max_leaf_nodes': 20, 'n_estimators': 64},
     {'max_depth': 9, 'max_leaf_nodes': 20, 'n_estimators': 100},
     {'max_depth': 9, 'max_leaf_nodes': 20, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 2, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 2, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 2, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 3, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 3, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 3, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 4, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 4, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 4, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 5, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 5, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 5, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 6, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 6, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 6, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 7, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 7, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 7, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 8, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 8, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 8, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 9, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 9, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 9, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 10, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 10, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 10, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 11, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 11, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 11, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 12, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 12, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 12, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 13, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 13, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 13, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 14, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 14, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 14, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 15, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 15, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 15, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 16, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 16, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 16, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 17, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 17, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 17, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 18, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 18, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 18, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 19, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 19, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 19, 'n_estimators': 128},
     {'max_depth': 10, 'max_leaf_nodes': 20, 'n_estimators': 64},
     {'max_depth': 10, 'max_leaf_nodes': 20, 'n_estimators': 100},
     {'max_depth': 10, 'max_leaf_nodes': 20, 'n_estimators': 128}]




```python
plt.plot(grid_model.cv_results_['mean_test_score']);
```


    
![png](output_72_0.png)
    



```python
y_pred = grid_model.predict(X_test)
display_evaluation(y_test, y_pred)
save_results('Gradient Boosting', y_test, y_pred)
```

                  precision    recall  f1-score   support
    
               0       0.88      0.82      0.85        44
               1       0.84      0.89      0.87        47
    
        accuracy                           0.86        91
       macro avg       0.86      0.86      0.86        91
    weighted avg       0.86      0.86      0.86        91
    



    
![png](output_73_1.png)
    


### 3.8 XGBoost


```python
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

#ns = [10, 20, 50, 64, 100, 128, 150, 200, 256]
ns = range(1, 31)
accs = []
for n in ns:
    
    model = XGBClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accs.append(acc)

```


```python
model.get_params()
```




    {'base_score': 0.5,
     'booster': 'gbtree',
     'colsample_bylevel': 1,
     'colsample_bynode': 1,
     'colsample_bytree': 1,
     'gamma': 0,
     'learning_rate': 0.1,
     'max_delta_step': 0,
     'max_depth': 3,
     'min_child_weight': 1,
     'missing': None,
     'n_estimators': 100,
     'n_jobs': 1,
     'nthread': None,
     'objective': 'binary:logistic',
     'random_state': 0,
     'reg_alpha': 0,
     'reg_lambda': 1,
     'scale_pos_weight': 1,
     'seed': None,
     'silent': None,
     'subsample': 1,
     'verbosity': 1}




```python
plt.plot(ns, accs, 'o-')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy');
```


    
![png](output_77_0.png)
    



```python
model = XGBClassifier(n_estimators=20)
model.fit(X_train, y_train)
```




    XGBClassifier(n_estimators=20)




```python
y_pred = model.predict(X_test)
display_evaluation(y_test, y_pred)
save_results('XGBoost', y_test, y_pred)
```

                  precision    recall  f1-score   support
    
               0       0.92      0.77      0.84        44
               1       0.81      0.94      0.87        47
    
        accuracy                           0.86        91
       macro avg       0.87      0.85      0.86        91
    weighted avg       0.87      0.86      0.86        91
    



    
![png](output_79_1.png)
    


## 4. Compare the results


```python
res
```




    {'Adaboost': 0.8681318681318682,
     'Decision Tree': 0.8021978021978022,
     'Gradient Boosting': 0.8571428571428571,
     'KNN': 0.8791208791208791,
     'Logistic Regression': 0.8461538461538461,
     'Random Forest': 0.8571428571428571,
     'SVM': 0.8571428571428571,
     'XGBoost': 0.8571428571428571}




```python
res_df = pd.Series(res)
res_df
```




    Logistic Regression    0.846154
    KNN                    0.879121
    SVM                    0.857143
    Decision Tree          0.802198
    Random Forest          0.857143
    Adaboost               0.868132
    Gradient Boosting      0.857143
    XGBoost                0.857143
    dtype: float64




```python
plt.figure()
sns.barplot(x=res_df.index, y=res_df)
plt.ylim(0.75, 0.9)
plt.xticks(rotation=90)
plt.ylabel('Accuracy Score');
```


    
![png](output_83_0.png)
    


## 5. Save and load a final model: KNN


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)

from sklearn.pipeline import Pipeline
operations = [('scaler', scaler), ('model', model)]
final_model = Pipeline(operations)
final_model.fit(X_train, y_train)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('model', KNeighborsClassifier(n_neighbors=7))])




```python
from joblib import dump, load

dump(final_model, mypath + 'heart_knn.joblib')
```




    ['/content/drive/MyDrive/Udemy/2022_PythonForMLDSMasterClass/project/heart_knn.joblib']




```python
patient = [[ 54. ,   1. ,   0. , 122. , 286. ,   0. ,   0. , 116. ,   1. ,
          3.2,   1. ,   2. ,   2. ]]
```


```python
X_test = pd.DataFrame(data=patient, columns=X.columns)
X_test
```





  <div id="df-1659288e-66e0-4e45-9c9a-e817fe8186af">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>54.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>122.0</td>
      <td>286.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>116.0</td>
      <td>1.0</td>
      <td>3.2</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1659288e-66e0-4e45-9c9a-e817fe8186af')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1659288e-66e0-4e45-9c9a-e817fe8186af button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1659288e-66e0-4e45-9c9a-e817fe8186af');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
loaded_model = load(mypath + 'heart_knn.joblib')
y_prob_pred = loaded_model.predict_proba(X_test)
y_pred = loaded_model.predict(X_test)
```


```python
print(y_prob_pred, y_pred)
```

    [[1. 0.]] [0]

