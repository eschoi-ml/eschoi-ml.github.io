# Evaluation of Binary Classification and Multi-Class Classification

1. Import libraries and Set up directories
2. Logistic Regression for Binary Classification: Hearing test data

    *   Read data and Quick data check
    *   Exploratory Data Analyasis (EDA)
    *   Modling
    *   Evaluating: Confusion Matrix, Reciever Operating Characteristic (ROC) curve and Precision-Recall curve for Binary Classification
    
        -  ROC curve in depth: y_pred vs. y_pred_prob vs. y_score

3. Logistic Regression for Multi-Class Classification: Iris flower data
    *   Read data and Quick data check
    *   Exploratory Data Analysis (EDA)
    *   Modeling
    *   Evaluating: Confusion Matrix and Receiver Operating Characteristic (ROC) curve for Multi-Class Classification



# 1. Import libraries and Set up directories


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```


```python
input_dir = '/DATA/'
output_dir = '/'
```

# 2. Logistic Regression for Binary Classification: Hearing test data

## 2.1 Read data and Quick data check


```python
df = pd.read_csv(input_dir + 'hearing_test.csv')
```


```python
df.head()
```





  <div id="df-a7e5d1c3-065f-4daa-a074-acf66f88ab8b" class="colab-df-container">
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
      <th>physical_score</th>
      <th>test_result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33.0</td>
      <td>40.7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50.0</td>
      <td>37.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>52.0</td>
      <td>24.7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56.0</td>
      <td>31.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>42.9</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-a7e5d1c3-065f-4daa-a074-acf66f88ab8b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
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

    .colab-df-buttons div {
      margin-bottom: 4px;
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
        document.querySelector('#df-a7e5d1c3-065f-4daa-a074-acf66f88ab8b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-a7e5d1c3-065f-4daa-a074-acf66f88ab8b');
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


<div id="df-c57d815e-1f4b-421e-bc64-0f11a40544dd">
  <button class="colab-df-quickchart" onclick="quickchart('df-c57d815e-1f4b-421e-bc64-0f11a40544dd')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-c57d815e-1f4b-421e-bc64-0f11a40544dd button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5000 entries, 0 to 4999
    Data columns (total 3 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   age             5000 non-null   float64
     1   physical_score  5000 non-null   float64
     2   test_result     5000 non-null   int64  
    dtypes: float64(2), int64(1)
    memory usage: 117.3 KB



```python
df.describe()
```





  <div id="df-8b0a1a41-3c70-48ea-bb32-5d58a006bf11" class="colab-df-container">
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
      <th>physical_score</th>
      <th>test_result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>51.609000</td>
      <td>32.760260</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.287001</td>
      <td>8.169802</td>
      <td>0.489947</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>43.000000</td>
      <td>26.700000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>51.000000</td>
      <td>35.300000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>60.000000</td>
      <td>38.900000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>50.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8b0a1a41-3c70-48ea-bb32-5d58a006bf11')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
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

    .colab-df-buttons div {
      margin-bottom: 4px;
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
        document.querySelector('#df-8b0a1a41-3c70-48ea-bb32-5d58a006bf11 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8b0a1a41-3c70-48ea-bb32-5d58a006bf11');
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


<div id="df-4dda96fb-e672-42c9-aa2f-c4921f14f0db">
  <button class="colab-df-quickchart" onclick="quickchart('df-4dda96fb-e672-42c9-aa2f-c4921f14f0db')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4dda96fb-e672-42c9-aa2f-c4921f14f0db button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




## 2.2 Exploratory Data Analysis (EDA)


```python
df['test_result'].value_counts()
```




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
      <th>count</th>
    </tr>
    <tr>
      <th>test_result</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
sns.boxplot(data=df, x='test_result', y='age');
```


    
![png](output_12_0.png)
    



```python
sns.boxplot(data=df, x='test_result', y='physical_score');
```


    
![png](output_13_0.png)
    



```python
sns.scatterplot(data=df, x='age', y='physical_score', hue='test_result');
```


    
![png](output_14_0.png)
    



```python
sns.heatmap(df.corr(), annot=True);
```


    
![png](output_15_0.png)
    



```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['age'],df['physical_score'],df['test_result'],c=df['test_result']);
```


    
![png](output_16_0.png)
    


## 2.3 Modeling


```python
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score

res = {}

def report_evaluation(model_name, y_test, y_pred):

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False, ax=axes[0])
    axes[0].set_title('Count')
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='true', colorbar=False, ax=axes[1])
    axes[1].set_title('Normalized over the true condition')
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='pred', colorbar=False, ax=axes[2]);
    axes[2].set_title('Normalized over the predicted condition')
    print(classification_report(y_test, y_pred))
    plt.tight_layout()

    res[model_name] = {'accuracy': accuracy_score(y_test, y_pred)}

```


```python
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

def plot_binaryclass_curves(model, X_test, y_pred):

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_pred, ax=axes[0]);
    PrecisionRecallDisplay.from_estimator(model, X_test, y_pred, ax=axes[1])
    plt.tight_layout();
```


```python
X = df.drop('test_result', axis=1)
y = df['test_result']
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)
```


```python
from sklearn.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(Cs=10, penalty='l2', solver='saga')
model.fit(scaled_X_train,y_train)

y_pred = model.predict(scaled_X_test)
y_pred_prob = model.predict_proba(scaled_X_test)
```


```python
model.coef_
```




    array([[-0.96352689,  3.35445967]])




```python
model.Cs
```




    10




```python
X_train.iloc[0]
```




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
      <th>141</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>32.0</td>
    </tr>
    <tr>
      <th>physical_score</th>
      <td>43.0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>




```python
scaled_X_train[0]
```




    array([-1.75380782,  1.25931551])




```python
y_train.iloc[0]
```




    np.int64(1)




```python
model.predict_proba(scaled_X_train[0].reshape(1, -1))
```




    array([[0.00163818, 0.99836182]])




```python
model.predict(scaled_X_train[0].reshape(1, -1))
```




    array([1])



## 2.4 Evaluating: Confusion Matrix, Receiver Operating Characteristic (ROC) curve and Preceision-Recall curve for Binary Classification


```python
report_evaluation('Logistic Regression', y_test, y_pred)
```

                  precision    recall  f1-score   support
    
               0       0.93      0.89      0.91       193
               1       0.93      0.96      0.95       307
    
        accuracy                           0.93       500
       macro avg       0.93      0.92      0.93       500
    weighted avg       0.93      0.93      0.93       500
    



    
![png](output_31_1.png)
    



```python
plot_binaryclass_curves(model, scaled_X_test, y_test)
```


    
![png](output_32_0.png)
    


### **ROC curve in depth: y_pred vs. y_pred_prob vs. y_score**
*   y_pred = model.predict(scaled_X_test)
*   y_pred_prob = model.predict_proba(scaled_X_test)
*   y_score = model.decision_function(scaled_X_test)



```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print("y_pred: ", y_pred)
print("---------------------------------------------------------------------")
print("fpr: ", fpr)
print("tpr: ", tpr)
print("thresholds: ", thresholds)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve using y_pred');
```

    y_pred:  [1 1 0 1 0 0 1 1 0 1 1 1 1 0 1 1 0 1 1 0 0 1 0 1 1 0 1 1 0 1 1 1 1 1 1 0 1
     1 0 1 1 1 1 0 0 1 1 1 1 1 1 0 0 0 1 0 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 1 0
     1 1 1 0 0 1 1 1 1 1 0 0 1 0 0 1 1 1 0 1 0 0 0 1 1 1 1 0 0 0 1 0 1 0 0 1 1
     0 0 1 0 0 0 1 1 0 1 0 0 1 1 0 1 0 1 0 1 1 1 1 0 1 1 0 0 0 0 1 0 1 1 1 0 0
     0 0 1 1 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 0 0 1 1 0 1 1 1 1 1 0 1 0 0 1 1 1
     0 1 1 0 1 1 0 0 0 1 1 1 1 0 1 1 0 1 0 1 1 1 0 1 0 1 1 0 1 0 1 1 0 0 0 1 1
     0 1 1 0 0 0 1 0 0 1 0 0 1 1 1 1 1 0 1 1 1 0 0 1 0 1 1 1 0 1 1 0 1 0 1 1 1
     1 1 0 0 1 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 0 0 0 0 0 1 1 1 0
     1 0 1 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 1 0 1 1 1 0 1 1 1 0 0 0
     1 0 1 1 1 1 1 0 0 1 1 1 1 1 0 1 1 0 0 1 1 1 0 1 1 0 1 0 1 0 1 1 1 1 0 1 1
     1 0 0 1 1 0 1 1 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 0 0 1 1 1 0 0 0 1 1 0
     1 0 0 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 1 1 1 1 1 1 0 0 1 0 1 1
     1 1 1 0 1 0 1 0 0 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 1 0 1
     1 0 0 1 0 1 1 0 1 0 0 1 1 1 1 1 0 0 0]
    ---------------------------------------------------------------------
    fpr:  [0.         0.10880829 1.        ]
    tpr:  [0.         0.95765472 1.        ]
    thresholds:  [inf  1.  0.]



    
![png](output_34_1.png)
    



```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
print("y_pred_prob: ", y_pred_prob)
print("---------------------------------------------------------------------")
print("fpr: ", fpr)
print("tpr: ", tpr)
print("thresholds: ", thresholds)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve using y_pred_prob');
```

    y_pred_prob:  [[2.47922017e-02 9.75207798e-01]
     [2.85166811e-02 9.71483319e-01]
     [9.87518098e-01 1.24819019e-02]
     [2.15353778e-03 9.97846462e-01]
     [9.72740911e-01 2.72590886e-02]
     [9.87894644e-01 1.21053564e-02]
     [7.61468420e-02 9.23853158e-01]
     [1.79611229e-02 9.82038877e-01]
     [9.96600711e-01 3.39928881e-03]
     [3.50346004e-02 9.64965400e-01]
     [8.63633662e-02 9.13636634e-01]
     [1.04910823e-02 9.89508918e-01]
     [7.56985228e-03 9.92430148e-01]
     [9.25707663e-01 7.42923365e-02]
     [1.36942875e-04 9.99863057e-01]
     [6.68388825e-02 9.33161118e-01]
     [9.88088637e-01 1.19113630e-02]
     [3.05375721e-03 9.96946243e-01]
     [6.42747617e-04 9.99357252e-01]
     [9.96008216e-01 3.99178424e-03]
     [8.76817020e-01 1.23182980e-01]
     [1.34524081e-01 8.65475919e-01]
     [5.48849598e-01 4.51150402e-01]
     [3.66915477e-02 9.63308452e-01]
     [4.07699245e-01 5.92300755e-01]
     [5.36114910e-01 4.63885090e-01]
     [4.05183256e-02 9.59481674e-01]
     [2.08877130e-02 9.79112287e-01]
     [9.07356597e-01 9.26434029e-02]
     [3.37496385e-04 9.99662504e-01]
     [4.75794506e-02 9.52420549e-01]
     [2.38706909e-03 9.97612931e-01]
     [1.90779505e-02 9.80922049e-01]
     [6.74152895e-03 9.93258471e-01]
     [4.57485017e-02 9.54251498e-01]
     [9.66170347e-01 3.38296527e-02]
     [6.79017138e-02 9.32098286e-01]
     [2.95821511e-02 9.70417849e-01]
     [9.86729981e-01 1.32700185e-02]
     [4.80415012e-02 9.51958499e-01]
     [6.69637175e-03 9.93303628e-01]
     [3.90527395e-01 6.09472605e-01]
     [3.73490512e-01 6.26509488e-01]
     [5.74249887e-01 4.25750113e-01]
     [9.99617792e-01 3.82208454e-04]
     [6.57568358e-02 9.34243164e-01]
     [6.72621490e-02 9.32737851e-01]
     [1.74222002e-02 9.82577800e-01]
     [5.81934581e-02 9.41806542e-01]
     [1.41841964e-01 8.58158036e-01]
     [1.84268583e-01 8.15731417e-01]
     [6.12324958e-01 3.87675042e-01]
     [5.71076275e-01 4.28923725e-01]
     [8.64506719e-01 1.35493281e-01]
     [5.18307957e-03 9.94816920e-01]
     [7.90555029e-01 2.09444971e-01]
     [3.99738164e-02 9.60026184e-01]
     [4.54020760e-03 9.95459792e-01]
     [2.13183727e-03 9.97868163e-01]
     [1.92720153e-01 8.07279847e-01]
     [3.07427192e-01 6.92572808e-01]
     [8.05985355e-01 1.94014645e-01]
     [8.13399819e-02 9.18660018e-01]
     [9.97618674e-01 2.38132602e-03]
     [2.47039705e-03 9.97529603e-01]
     [6.01075039e-02 9.39892496e-01]
     [9.95596054e-01 4.40394596e-03]
     [3.63344915e-02 9.63665508e-01]
     [1.35710154e-01 8.64289846e-01]
     [4.57485017e-02 9.54251498e-01]
     [5.51291587e-03 9.94487084e-01]
     [3.10912639e-02 9.68908736e-01]
     [3.48065672e-02 9.65193433e-01]
     [9.85454916e-01 1.45450836e-02]
     [1.16506305e-02 9.88349370e-01]
     [1.66204204e-02 9.83379580e-01]
     [1.73665847e-01 8.26334153e-01]
     [9.95756252e-01 4.24374811e-03]
     [9.61394210e-01 3.86057900e-02]
     [1.71920446e-02 9.82807955e-01]
     [1.52428674e-03 9.98475713e-01]
     [1.33387192e-02 9.86661281e-01]
     [1.18291869e-01 8.81708131e-01]
     [4.62662689e-01 5.37337311e-01]
     [9.83746768e-01 1.62532318e-02]
     [9.85559618e-01 1.44403821e-02]
     [1.75698174e-01 8.24301826e-01]
     [8.13456061e-01 1.86543939e-01]
     [9.90603004e-01 9.39699571e-03]
     [4.28272140e-03 9.95717279e-01]
     [3.78385195e-01 6.21614805e-01]
     [3.21402472e-03 9.96785975e-01]
     [9.99955188e-01 4.48117833e-05]
     [9.56738445e-04 9.99043262e-01]
     [8.37624483e-01 1.62375517e-01]
     [5.87564759e-01 4.12435241e-01]
     [9.56680972e-01 4.33190281e-02]
     [3.86662640e-03 9.96133374e-01]
     [9.25146226e-02 9.07485377e-01]
     [1.09640918e-01 8.90359082e-01]
     [3.28156227e-03 9.96718438e-01]
     [8.92177652e-01 1.07822348e-01]
     [9.97124982e-01 2.87501827e-03]
     [9.73796999e-01 2.62030006e-02]
     [1.88457995e-01 8.11542005e-01]
     [9.51107698e-01 4.88923018e-02]
     [3.95861320e-02 9.60413868e-01]
     [9.99571942e-01 4.28057788e-04]
     [9.94573830e-01 5.42617042e-03]
     [2.16745011e-02 9.78325499e-01]
     [1.52676210e-02 9.84732379e-01]
     [9.99857728e-01 1.42272201e-04]
     [9.73811395e-01 2.61886047e-02]
     [3.54565198e-01 6.45434802e-01]
     [9.43940869e-01 5.60591313e-02]
     [8.38007509e-01 1.61992491e-01]
     [9.98509894e-01 1.49010588e-03]
     [4.48956607e-01 5.51043393e-01]
     [1.57997453e-01 8.42002547e-01]
     [9.85111534e-01 1.48884655e-02]
     [1.60140932e-02 9.83985907e-01]
     [9.99860979e-01 1.39021342e-04]
     [9.97972618e-01 2.02738228e-03]
     [2.84639022e-01 7.15360978e-01]
     [1.83836466e-02 9.81616353e-01]
     [7.32959127e-01 2.67040873e-01]
     [2.45377876e-03 9.97546221e-01]
     [9.96023880e-01 3.97612022e-03]
     [1.54872359e-01 8.45127641e-01]
     [7.55800602e-01 2.44199398e-01]
     [1.21619070e-03 9.98783809e-01]
     [1.91782435e-03 9.98082176e-01]
     [4.81964627e-02 9.51803537e-01]
     [3.05270430e-01 6.94729570e-01]
     [9.11626172e-01 8.83738279e-02]
     [2.32760853e-01 7.67239147e-01]
     [1.77850364e-03 9.98221496e-01]
     [6.07360884e-01 3.92639116e-01]
     [7.70573642e-01 2.29426358e-01]
     [9.91048846e-01 8.95115426e-03]
     [9.99597676e-01 4.02324128e-04]
     [4.78870129e-02 9.52112987e-01]
     [7.45458002e-01 2.54541998e-01]
     [1.24699682e-02 9.87530032e-01]
     [1.51914634e-03 9.98480854e-01]
     [1.88975970e-01 8.11024030e-01]
     [9.92823276e-01 7.17672403e-03]
     [9.34356738e-01 6.56432622e-02]
     [9.99891399e-01 1.08601255e-04]
     [5.02768635e-01 4.97231365e-01]
     [4.72769563e-01 5.27230437e-01]
     [1.24116675e-01 8.75883325e-01]
     [9.75334015e-01 2.46659849e-02]
     [3.54060243e-04 9.99645940e-01]
     [1.74720363e-01 8.25279637e-01]
     [4.78870129e-02 9.52112987e-01]
     [5.31905528e-01 4.68094472e-01]
     [5.43615927e-02 9.45638407e-01]
     [9.99354451e-01 6.45549010e-04]
     [5.56215755e-02 9.44378425e-01]
     [2.98749097e-02 9.70125090e-01]
     [9.93676536e-01 6.32346364e-03]
     [9.47224855e-03 9.90527751e-01]
     [8.07479827e-01 1.92520173e-01]
     [4.40462941e-01 5.59537059e-01]
     [9.77012082e-01 2.29879176e-02]
     [1.43151097e-01 8.56848903e-01]
     [7.67219796e-03 9.92327802e-01]
     [5.62492319e-01 4.37507681e-01]
     [9.97723470e-01 2.27653039e-03]
     [1.88263409e-02 9.81173659e-01]
     [4.42701630e-02 9.55729837e-01]
     [9.74168518e-01 2.58314822e-02]
     [4.72483552e-02 9.52751645e-01]
     [1.63371213e-02 9.83662879e-01]
     [6.31059907e-02 9.36894009e-01]
     [2.64585756e-03 9.97354142e-01]
     [3.21443138e-02 9.67855686e-01]
     [9.53414261e-01 4.65857389e-02]
     [4.86560715e-01 5.13439285e-01]
     [6.40422856e-01 3.59577144e-01]
     [9.90486833e-01 9.51316668e-03]
     [1.18074925e-02 9.88192507e-01]
     [1.22156426e-02 9.87784357e-01]
     [2.63546107e-03 9.97364539e-01]
     [9.81527428e-01 1.84725719e-02]
     [4.02845293e-03 9.95971547e-01]
     [1.65652154e-02 9.83434785e-01]
     [5.91794204e-01 4.08205796e-01]
     [1.46691077e-02 9.85330892e-01]
     [1.94304135e-01 8.05695865e-01]
     [9.99863925e-01 1.36074518e-04]
     [7.91114658e-01 2.08885342e-01]
     [6.60798717e-01 3.39201283e-01]
     [1.58072543e-01 8.41927457e-01]
     [3.69670898e-03 9.96303291e-01]
     [1.12778270e-03 9.98872217e-01]
     [2.41951350e-03 9.97580487e-01]
     [6.32464412e-01 3.67535588e-01]
     [9.89946377e-03 9.90100536e-01]
     [1.48702343e-01 8.51297657e-01]
     [9.84129935e-01 1.58700645e-02]
     [2.50473945e-04 9.99749526e-01]
     [9.99246677e-01 7.53322703e-04]
     [1.26714215e-01 8.73285785e-01]
     [1.88975970e-01 8.11024030e-01]
     [2.82368450e-02 9.71763155e-01]
     [9.99594945e-01 4.05054522e-04]
     [8.21441623e-02 9.17855838e-01]
     [9.89850311e-01 1.01496887e-02]
     [6.81540745e-04 9.99318459e-01]
     [2.21946502e-02 9.77805350e-01]
     [8.40366468e-01 1.59633532e-01]
     [4.11927095e-01 5.88072905e-01]
     [5.58185932e-01 4.41814068e-01]
     [3.98744345e-01 6.01255655e-01]
     [1.26340313e-01 8.73659687e-01]
     [9.51551506e-01 4.84484945e-02]
     [8.68098525e-01 1.31901475e-01]
     [9.99162031e-01 8.37968526e-04]
     [1.64553457e-02 9.83544654e-01]
     [1.18536413e-02 9.88146359e-01]
     [9.89386945e-01 1.06130546e-02]
     [1.64556197e-01 8.35443803e-01]
     [1.15729735e-02 9.88427027e-01]
     [9.78661089e-01 2.13389105e-02]
     [7.52559937e-01 2.47440063e-01]
     [9.86906016e-01 1.30939843e-02]
     [2.99895306e-02 9.70010469e-01]
     [9.96336951e-01 3.66304886e-03]
     [9.86551613e-01 1.34483871e-02]
     [2.50387798e-02 9.74961220e-01]
     [7.76010362e-01 2.23989638e-01]
     [8.99590827e-01 1.00409173e-01]
     [1.61449001e-04 9.99838551e-01]
     [1.93247052e-01 8.06752948e-01]
     [6.12686492e-04 9.99387314e-01]
     [3.01704754e-02 9.69829525e-01]
     [7.17086488e-02 9.28291351e-01]
     [9.36031186e-01 6.39688136e-02]
     [6.49321256e-03 9.93506787e-01]
     [6.96714473e-02 9.30328553e-01]
     [1.41801592e-02 9.85819841e-01]
     [5.22488928e-01 4.77511072e-01]
     [9.98392223e-01 1.60777667e-03]
     [3.04553397e-01 6.95446603e-01]
     [9.96868507e-01 3.13149315e-03]
     [1.38451764e-02 9.86154824e-01]
     [3.78709344e-03 9.96212907e-01]
     [1.51241796e-02 9.84875820e-01]
     [9.99961451e-01 3.85487815e-05]
     [5.43615927e-02 9.45638407e-01]
     [2.55391244e-02 9.74460876e-01]
     [9.45034716e-01 5.49652836e-02]
     [5.06238870e-03 9.94937611e-01]
     [9.95727563e-01 4.27243711e-03]
     [2.77539807e-03 9.97224602e-01]
     [2.73522221e-01 7.26477779e-01]
     [1.75208731e-01 8.24791269e-01]
     [2.51215023e-02 9.74878498e-01]
     [5.68557913e-03 9.94314421e-01]
     [8.37086867e-01 1.62913133e-01]
     [9.99238569e-01 7.61430917e-04]
     [1.53988798e-01 8.46011202e-01]
     [1.53270837e-02 9.84672916e-01]
     [9.96359517e-01 3.64048329e-03]
     [2.49563227e-02 9.75043677e-01]
     [3.28328873e-01 6.71671127e-01]
     [9.89638547e-01 1.03614531e-02]
     [9.93840896e-01 6.15910448e-03]
     [3.70519778e-02 9.62948022e-01]
     [9.88721775e-01 1.12782246e-02]
     [9.02209171e-02 9.09779083e-01]
     [1.25116991e-02 9.87488301e-01]
     [6.89801964e-02 9.31019804e-01]
     [6.65930910e-02 9.33406909e-01]
     [9.23027232e-01 7.69727679e-02]
     [2.01957353e-02 9.79804265e-01]
     [6.14972813e-03 9.93850272e-01]
     [2.41328804e-02 9.75867120e-01]
     [1.23869184e-02 9.87613082e-01]
     [3.11933426e-02 9.68806657e-01]
     [5.80083132e-02 9.41991687e-01]
     [1.30306073e-02 9.86969393e-01]
     [9.96857928e-01 3.14207215e-03]
     [3.75207445e-01 6.24792555e-01]
     [2.52877468e-02 9.74712253e-01]
     [9.93202670e-01 6.79732990e-03]
     [9.99463396e-01 5.36604389e-04]
     [9.99704590e-01 2.95409989e-04]
     [9.43761574e-01 5.62384264e-02]
     [7.15048737e-01 2.84951263e-01]
     [8.70098728e-03 9.91299013e-01]
     [1.51402324e-03 9.98485977e-01]
     [1.64556197e-01 8.35443803e-01]
     [5.64988462e-01 4.35011538e-01]
     [2.93096984e-03 9.97069030e-01]
     [9.95047958e-01 4.95204213e-03]
     [7.04646001e-03 9.92953540e-01]
     [4.64850748e-03 9.95351493e-01]
     [1.43983057e-01 8.56016943e-01]
     [1.26714215e-01 8.73285785e-01]
     [4.98539693e-01 5.01460307e-01]
     [1.45659040e-01 8.54340960e-01]
     [9.18545397e-01 8.14546032e-02]
     [9.99617576e-01 3.82424125e-04]
     [3.16260102e-01 6.83739898e-01]
     [2.86095933e-03 9.97139041e-01]
     [1.02081025e-02 9.89791898e-01]
     [1.25116991e-02 9.87488301e-01]
     [2.33359575e-02 9.76664043e-01]
     [1.98576317e-01 8.01423683e-01]
     [9.99452089e-01 5.47911191e-04]
     [5.80149408e-03 9.94198506e-01]
     [2.75656309e-01 7.24343691e-01]
     [1.22565329e-02 9.87743467e-01]
     [1.25967354e-01 8.74032646e-01]
     [2.16028777e-02 9.78397122e-01]
     [6.87662620e-01 3.12337380e-01]
     [9.99892495e-01 1.07504695e-04]
     [9.99400727e-01 5.99273140e-04]
     [9.42811785e-02 9.05718822e-01]
     [9.97666414e-01 2.33358628e-03]
     [3.61563776e-01 6.38436224e-01]
     [3.72831151e-01 6.27168849e-01]
     [9.36570534e-02 9.06342947e-01]
     [7.06003267e-01 2.93996733e-01]
     [1.76553815e-02 9.82344618e-01]
     [3.03152053e-03 9.96968479e-01]
     [1.53988798e-01 8.46011202e-01]
     [9.41350010e-01 5.86499895e-02]
     [9.65349950e-01 3.46500495e-02]
     [9.92186061e-01 7.81393862e-03]
     [7.68639255e-02 9.23136075e-01]
     [9.97459546e-01 2.54045375e-03]
     [5.87522244e-02 9.41247776e-01]
     [4.12746891e-01 5.87253109e-01]
     [2.77611309e-02 9.72238869e-01]
     [1.85799089e-01 8.14200911e-01]
     [6.65930910e-02 9.33406909e-01]
     [9.99886836e-01 1.13164316e-04]
     [9.98871126e-01 1.12887439e-03]
     [2.41328804e-02 9.75867120e-01]
     [2.54893239e-01 7.45106761e-01]
     [8.81849020e-03 9.91181510e-01]
     [1.39378654e-02 9.86062135e-01]
     [4.10477153e-02 9.58952285e-01]
     [9.92648433e-01 7.35156696e-03]
     [5.22101564e-03 9.94778984e-01]
     [4.53074550e-02 9.54692545e-01]
     [6.71332978e-01 3.28667022e-01]
     [9.80638059e-01 1.93619414e-02]
     [1.06361663e-03 9.98936383e-01]
     [3.62162175e-02 9.63783782e-01]
     [1.24283748e-02 9.87571625e-01]
     [9.99975899e-01 2.41005747e-05]
     [2.87062703e-03 9.97129373e-01]
     [4.27858092e-01 5.72141908e-01]
     [9.99860114e-01 1.39886114e-04]
     [7.56722018e-02 9.24327798e-01]
     [9.99073641e-01 9.26358779e-04]
     [3.64531372e-02 9.63546863e-01]
     [8.72678088e-01 1.27321912e-01]
     [2.94498155e-01 7.05501845e-01]
     [7.27166244e-04 9.99272834e-01]
     [5.86361828e-03 9.94136382e-01]
     [4.09689590e-03 9.95903104e-01]
     [7.00235899e-01 2.99764101e-01]
     [5.49149913e-02 9.45085009e-01]
     [9.66461398e-03 9.90335386e-01]
     [2.81207197e-01 7.18792803e-01]
     [9.82259152e-01 1.77408477e-02]
     [9.96260013e-01 3.73998652e-03]
     [2.80409977e-01 7.19590023e-01]
     [7.09987280e-02 9.29001272e-01]
     [9.83961733e-01 1.60382668e-02]
     [1.60674915e-02 9.83932508e-01]
     [1.47180883e-02 9.85281912e-01]
     [2.47154718e-01 7.52845282e-01]
     [9.99882081e-01 1.17919079e-04]
     [4.93324233e-01 5.06675767e-01]
     [1.07870038e-03 9.98921300e-01]
     [4.51468828e-01 5.48531172e-01]
     [3.75582439e-02 9.62441756e-01]
     [9.99842176e-01 1.57823812e-04]
     [2.10265442e-02 9.78973456e-01]
     [2.67513741e-01 7.32486259e-01]
     [1.97979478e-02 9.80202052e-01]
     [5.04433547e-02 9.49556645e-01]
     [9.83308303e-01 1.66916967e-02]
     [4.75794506e-02 9.52420549e-01]
     [3.40011257e-02 9.65998874e-01]
     [1.51290369e-01 8.48709631e-01]
     [1.26723198e-03 9.98732768e-01]
     [3.22497349e-02 9.67750265e-01]
     [1.02168776e-01 8.97831224e-01]
     [9.11307628e-01 8.86923720e-02]
     [9.03248244e-01 9.67517562e-02]
     [3.38457789e-03 9.96615422e-01]
     [1.88784810e-02 9.81121519e-01]
     [5.55013699e-03 9.94449863e-01]
     [7.90555029e-01 2.09444971e-01]
     [9.69982279e-01 3.00177207e-02]
     [9.79287783e-01 2.07122172e-02]
     [6.36261149e-04 9.99363739e-01]
     [4.66428860e-02 9.53357114e-01]
     [9.30884990e-01 6.91150097e-02]
     [7.46920499e-02 9.25307950e-01]
     [9.28189241e-01 7.18107590e-02]
     [9.99907884e-01 9.21160783e-05]
     [1.73067465e-02 9.82693253e-01]
     [1.24699682e-02 9.87530032e-01]
     [8.77606615e-01 1.22393385e-01]
     [3.41389032e-01 6.58610968e-01]
     [6.98910555e-02 9.30108944e-01]
     [2.28558392e-01 7.71441608e-01]
     [3.84237393e-02 9.61576261e-01]
     [2.42927455e-02 9.75707254e-01]
     [9.12979527e-01 8.70204730e-02]
     [5.87522244e-02 9.41247776e-01]
     [1.19710624e-01 8.80289376e-01]
     [4.80415012e-02 9.51958499e-01]
     [9.97315598e-01 2.68440183e-03]
     [1.44819034e-01 8.55180966e-01]
     [4.29092165e-04 9.99570908e-01]
     [9.99505502e-01 4.94497752e-04]
     [9.99696145e-01 3.03854765e-04]
     [3.07869639e-02 9.69213036e-01]
     [9.99837298e-01 1.62702443e-04]
     [9.99045567e-01 9.54433329e-04]
     [5.15732558e-01 4.84267442e-01]
     [9.94305748e-01 5.69425186e-03]
     [2.83950642e-01 7.16049358e-01]
     [1.46926621e-01 8.53073379e-01]
     [3.59835299e-03 9.96401647e-01]
     [3.81745195e-02 9.61825480e-01]
     [7.66242125e-02 9.23375788e-01]
     [1.26723198e-03 9.98732768e-01]
     [9.98085714e-01 1.91428601e-03]
     [6.66838611e-01 3.33161389e-01]
     [1.84685315e-03 9.98153147e-01]
     [9.94212416e-01 5.78758389e-03]
     [2.93678650e-01 7.06321350e-01]
     [6.27504621e-03 9.93724954e-01]
     [4.13851627e-03 9.95861484e-01]
     [3.19344703e-02 9.68065530e-01]
     [1.92632369e-01 8.07367631e-01]
     [9.99857165e-01 1.42834852e-04]
     [4.22539817e-03 9.95774602e-01]
     [9.96386010e-01 3.61398970e-03]
     [3.22497349e-02 9.67750265e-01]
     [8.81253489e-01 1.18746511e-01]
     [6.92966610e-01 3.07033390e-01]
     [4.58631084e-03 9.95413689e-01]
     [1.67313734e-02 9.83268627e-01]
     [2.47784762e-01 7.52215238e-01]
     [9.97432252e-01 2.56774791e-03]
     [9.97560372e-01 2.43962815e-03]
     [8.24168261e-03 9.91758317e-01]
     [1.96670656e-02 9.80332934e-01]
     [9.98847353e-01 1.15264653e-03]
     [4.31167272e-03 9.95688327e-01]
     [2.62358321e-02 9.73764168e-01]
     [4.72769563e-01 5.27230437e-01]
     [3.79268525e-02 9.62073148e-01]
     [9.99944057e-03 9.90000559e-01]
     [1.35707409e-02 9.86429259e-01]
     [1.37072814e-02 9.86292719e-01]
     [9.78649301e-01 2.13506990e-02]
     [7.02282799e-03 9.92977172e-01]
     [7.18482845e-01 2.81517155e-01]
     [1.35710154e-01 8.64289846e-01]
     [1.24283748e-02 9.87571625e-01]
     [2.04088333e-02 9.79591167e-01]
     [1.17178986e-01 8.82821014e-01]
     [7.46467932e-03 9.92535321e-01]
     [4.49204139e-03 9.95507959e-01]
     [9.68226923e-01 3.17730770e-02]
     [8.42836855e-04 9.99157163e-01]
     [9.51733167e-01 4.82668333e-02]
     [1.18291869e-01 8.81708131e-01]
     [9.16003204e-03 9.90839968e-01]
     [7.16425401e-01 2.83574599e-01]
     [8.36239312e-01 1.63760688e-01]
     [2.03413054e-02 9.79658695e-01]
     [9.99731585e-01 2.68414787e-04]
     [2.88939861e-02 9.71106014e-01]
     [2.22681910e-02 9.77731809e-01]
     [9.98195565e-01 1.80443535e-03]
     [1.81739461e-01 8.18260539e-01]
     [9.34356738e-01 6.56432622e-02]
     [7.72363049e-01 2.27636951e-01]
     [5.45357723e-02 9.45464228e-01]
     [2.00501131e-03 9.97994989e-01]
     [7.93855143e-03 9.92061449e-01]
     [4.60448006e-02 9.53955199e-01]
     [1.94215804e-01 8.05784196e-01]
     [9.77857690e-01 2.21423097e-02]
     [9.93802815e-01 6.19718464e-03]
     [9.23546450e-01 7.64535497e-02]]
    ---------------------------------------------------------------------
    fpr:  [0.         0.         0.         0.00518135 0.00518135 0.01036269
     0.01036269 0.01036269 0.01036269 0.01554404 0.01554404 0.02072539
     0.02072539 0.02590674 0.02590674 0.02590674 0.02590674 0.03108808
     0.03108808 0.03108808 0.03108808 0.03108808 0.03108808 0.03626943
     0.03626943 0.03626943 0.03626943 0.03626943 0.03626943 0.03626943
     0.03626943 0.04145078 0.04145078 0.04145078 0.04145078 0.04663212
     0.04663212 0.05181347 0.05181347 0.05181347 0.05181347 0.05181347
     0.05181347 0.05181347 0.05699482 0.05699482 0.05699482 0.05699482
     0.05699482 0.05699482 0.05699482 0.05699482 0.06217617 0.06217617
     0.06735751 0.06735751 0.07253886 0.07253886 0.09326425 0.09326425
     0.10362694 0.10362694 0.10880829 0.10880829 0.10880829 0.11917098
     0.11917098 0.12435233 0.12435233 0.12953368 0.12953368 0.13471503
     0.13471503 0.14507772 0.14507772 0.16062176 0.16062176 0.22279793
     0.22279793 0.23316062 0.37305699 0.37305699 0.38341969 1.        ]
    tpr:  [0.         0.00325733 0.05537459 0.05863192 0.16938111 0.16938111
     0.29641694 0.31596091 0.36156352 0.36156352 0.41042345 0.41042345
     0.42019544 0.42019544 0.43648208 0.44299674 0.46905537 0.46905537
     0.50814332 0.51465798 0.5732899  0.57980456 0.58306189 0.58306189
     0.58631922 0.60586319 0.61237785 0.61889251 0.63517915 0.64169381
     0.64495114 0.64495114 0.64820847 0.65472313 0.66123779 0.66123779
     0.68403909 0.68403909 0.72312704 0.72964169 0.74267101 0.74918567
     0.752443   0.75895765 0.75895765 0.78175896 0.78827362 0.7980456
     0.80456026 0.83061889 0.83713355 0.84039088 0.84039088 0.88925081
     0.88925081 0.90228013 0.90228013 0.90879479 0.90879479 0.9218241
     0.9218241  0.94136808 0.94136808 0.94788274 0.95765472 0.95765472
     0.96091205 0.96091205 0.96742671 0.96742671 0.97394137 0.97394137
     0.98371336 0.98371336 0.98697068 0.98697068 0.99348534 0.99348534
     0.99674267 0.99674267 0.99674267 1.         1.         1.        ]
    thresholds:  [           inf 9.99863057e-01 9.98783809e-01 9.98732768e-01
     9.95688327e-01 9.95507959e-01 9.87613082e-01 9.87488301e-01
     9.83932508e-01 9.83662879e-01 9.80202052e-01 9.79804265e-01
     9.79112287e-01 9.78973456e-01 9.76664043e-01 9.75867120e-01
     9.73764168e-01 9.72238869e-01 9.67855686e-01 9.67750265e-01
     9.54692545e-01 9.54251498e-01 9.53955199e-01 9.53357114e-01
     9.52751645e-01 9.51958499e-01 9.49556645e-01 9.45638407e-01
     9.41806542e-01 9.41247776e-01 9.39892496e-01 9.36894009e-01
     9.34243164e-01 9.33406909e-01 9.32737851e-01 9.32098286e-01
     9.24327798e-01 9.23853158e-01 8.82821014e-01 8.81708131e-01
     8.73659687e-01 8.73285785e-01 8.65475919e-01 8.64289846e-01
     8.58158036e-01 8.48709631e-01 8.46011202e-01 8.41927457e-01
     8.35443803e-01 8.11542005e-01 8.11024030e-01 8.07367631e-01
     8.07279847e-01 7.16049358e-01 7.15360978e-01 6.94729570e-01
     6.92572808e-01 6.71671127e-01 6.27168849e-01 6.09472605e-01
     5.92300755e-01 5.48531172e-01 5.37337311e-01 5.27230437e-01
     5.01460307e-01 4.84267442e-01 4.77511072e-01 4.68094472e-01
     4.51150402e-01 4.41814068e-01 4.35011538e-01 4.28923725e-01
     4.08205796e-01 3.87675042e-01 3.67535588e-01 3.33161389e-01
     3.12337380e-01 2.27636951e-01 2.23989638e-01 2.09444971e-01
     7.18107590e-02 6.91150097e-02 6.56432622e-02 2.41005747e-05]



    
![png](output_35_1.png)
    



```python
y_score = model.decision_function(scaled_X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
print("y_score: ", y_score)
print("---------------------------------------------------------------------")
print("fpr: ", fpr)
print("tpr: ", tpr)
print("thresholds: ", thresholds)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve using y_score');
```

    y_score:  [ 3.67212142e+00  3.52833488e+00 -4.37091508e+00  6.13848745e+00
     -3.57473078e+00 -4.40192803e+00  2.49588953e+00  4.00142131e+00
     -5.68078396e+00  3.31575609e+00  2.35886935e+00  4.54668318e+00
      4.87598308e+00 -2.52255068e+00  8.89580973e+00  2.63629289e+00
     -4.41827959e+00  5.78832415e+00  7.34911547e+00 -5.51951720e+00
     -1.96262743e+00  1.86153633e+00 -1.96023678e-01  3.26782724e+00
      3.73484779e-01 -1.44711652e-01  3.16463886e+00  3.84748524e+00
     -2.28177779e+00  7.99361820e+00  2.99660573e+00  6.03529907e+00
      3.93995975e+00  4.99270418e+00  3.03776822e+00 -3.35200245e+00
      2.61937699e+00  3.49055557e+00 -4.30888918e+00  2.98645619e+00
      4.99947054e+00  4.45095881e-01  5.17271317e-01 -2.99212061e-01
     -7.86916213e+00  2.65377312e+00  2.62952653e+00  4.03243426e+00
      2.78402694e+00  1.80007477e+00  1.48769077e+00 -4.57095649e-01
     -2.86243675e-01 -1.85323702e+00  5.25715933e+00 -1.32827423e+00
      3.17873591e+00  5.39023200e+00  6.14863699e+00  1.43243123e+00
      8.12175083e-01 -1.42413192e+00  2.42427843e+00 -6.03771363e+00
      6.00090294e+00  2.74963081e+00 -5.42084066e+00  3.27797678e+00
      1.85138679e+00  3.03776822e+00  5.19513343e+00  3.43924355e+00
      3.32252245e+00 -4.21585034e+00  4.44067595e+00  4.08036311e+00
      1.55986621e+00 -5.45805563e+00 -3.21498227e+00  4.04596698e+00
      6.48470324e+00  4.30365577e+00  2.00870605e+00  1.49627782e-01
     -4.10307675e+00 -4.22318103e+00  1.54576915e+00 -1.47262510e+00
     -4.65792382e+00  5.44887471e+00  4.96407907e-01  5.73701212e+00
     -1.00129946e+01  6.95102332e+00 -1.64065823e+00 -3.53907266e-01
     -3.09487799e+00  5.55149876e+00  2.28331074e+00  2.09441420e+00
      5.71614871e+00 -2.11318033e+00 -5.84881709e+00 -3.61532893e+00
      1.46006100e+00 -2.96800735e+00  3.18888545e+00 -7.75582420e+00
     -5.21108071e+00  3.80970593e+00  4.16663559e+00 -8.85762614e+00
     -3.61589326e+00  5.99031958e-01 -2.82365648e+00 -1.64347708e+00
     -6.50741688e+00  2.04887320e-01  1.67320413e+00 -4.19216808e+00
      4.11814242e+00 -8.88074407e+00 -6.19898040e+00  9.21565493e-01
      3.97773906e+00 -1.00968821e+00  6.00766930e+00 -5.52346471e+00
      1.69688638e+00 -1.12979249e+00  6.71081475e+00  6.25464422e+00
      2.98307301e+00  8.22324622e-01 -2.33365415e+00  1.19278700e+00
      6.33020283e+00 -4.36232240e-01 -1.21155313e+00 -4.70698133e+00
     -7.81785010e+00  2.98983937e+00 -1.07453295e+00  4.37188370e+00
      6.48808642e+00  1.45667782e+00 -4.92970966e+00 -2.65562335e+00
     -9.12771899e+00 -1.10746532e-02  1.09029628e-01  1.95401084e+00
     -3.67735483e+00  7.94568936e+00  1.55253551e+00  2.98983937e+00
     -1.27795755e-01  2.85620237e+00 -7.34476367e+00  2.83195579e+00
      3.48040604e+00 -5.05714464e+00  4.64987156e+00 -1.43371713e+00
      2.39283447e-01 -3.74953026e+00  1.78936090e+00  4.86245036e+00
     -2.51283215e-01 -6.08282363e+00  3.95349247e+00  3.07216435e+00
     -3.62999031e+00  3.00393642e+00  4.09784334e+00  2.69775445e+00
      5.93211068e+00  3.40484742e+00 -3.01875504e+00  5.37700903e-02
     -5.77199930e-01 -4.64551977e+00  4.42714323e+00  4.39274711e+00
      5.93605820e+00 -3.97282293e+00  5.51033628e+00  4.08374629e+00
     -3.71387496e-01  4.20723375e+00  1.42228169e+00 -8.90217181e+00
     -1.33165741e+00 -6.66855596e-01  1.67263979e+00  5.59660876e+00
      6.78637337e+00  6.02176635e+00 -5.42803802e-01  4.60532590e+00
      1.74481523e+00 -4.12732334e+00  8.29190515e+00 -7.19026326e+00
      1.93032859e+00  1.45667782e+00  3.53848442e+00 -7.81108374e+00
      2.41356456e+00 -4.58011069e+00  7.29047275e+00  3.78545934e+00
     -1.66095731e+00  3.56004549e-01 -2.33802985e-01  4.10699754e-01
      1.93371177e+00 -2.97759255e+00 -1.88424997e+00 -7.08369170e+00
      4.09051265e+00  4.42319572e+00 -4.53500069e+00  1.62471095e+00
      4.44744231e+00 -3.82565321e+00 -1.11231226e+00 -4.32242190e+00
      3.47645852e+00 -5.60578968e+00 -4.29535646e+00  3.66197188e+00
     -1.24256608e+00 -2.19268646e+00  8.73115979e+00  1.42904805e+00
      7.39704431e+00  3.47025650e+00  2.56073427e+00 -2.68325312e+00
      5.03048348e+00  2.59174722e+00  4.24162988e+00 -9.00164473e-02
     -6.43129393e+00  8.25707802e-01 -5.76310894e+00  4.26587647e+00
      5.57236217e+00  4.17622080e+00 -1.01635475e+01  2.85620237e+00
      3.64167280e+00 -2.84451989e+00  5.28084159e+00 -5.45128927e+00
      5.88418184e+00  9.76825031e-01  1.54915233e+00  3.65858870e+00
      5.16412048e+00 -1.63671072e+00 -7.17954939e+00  1.70365274e+00
      4.16268808e+00 -5.61199171e+00  3.66535506e+00  7.15753059e-01
     -4.55924728e+00 -5.08364574e+00  3.25767770e+00 -4.47353913e+00
      2.31094051e+00  4.36850052e+00  2.60246109e+00  2.64024040e+00
     -2.48420704e+00  3.88188137e+00  5.08517869e+00  3.69975119e+00
      4.37865005e+00  3.43586037e+00  2.78741012e+00  4.32733803e+00
     -5.75972576e+00  5.09940625e-01  3.65182234e+00 -4.98440487e+00
     -7.52971269e+00 -8.12685092e+00 -2.82027330e+00 -9.20032543e-01
      4.73557972e+00  6.49146960e+00  1.62471095e+00 -2.61432754e-01
      5.82948663e+00 -5.30299089e+00  4.94815851e+00  5.36654974e+00
      1.78259454e+00  1.93032859e+00  5.84124443e-03  1.76906182e+00
     -2.42274548e+00 -7.86859779e+00  7.71012597e-01  5.85373322e+00
      4.57431295e+00  4.36850052e+00  3.73414732e+00  1.39521626e+00
     -7.50884928e+00  5.14382141e+00  9.66111160e-01  4.38936393e+00
      1.93709495e+00  3.81308911e+00 -7.89214391e-01 -9.13786853e+00
     -7.41919362e+00  2.26244733e+00 -6.05801271e+00  5.68583342e-01
      5.20090164e-01  2.26977802e+00 -8.76051209e-01  4.01890154e+00
      5.79565484e+00  1.70365274e+00 -2.77572763e+00 -3.32719152e+00
     -4.84400151e+00  2.48573999e+00 -5.97286888e+00  2.77387740e+00
      3.52621369e-01  3.55596465e+00  1.47754123e+00  2.64024040e+00
     -9.08655650e+00 -6.78540475e+00  3.69975119e+00  1.07268272e+00
      4.72204700e+00  4.25911011e+00  3.15110614e+00 -4.90546307e+00
      5.24982864e+00  3.04791776e+00 -7.14220109e-01 -3.92489408e+00
      6.84501608e+00  3.28135996e+00  4.37526687e+00 -1.06332508e+01
      5.85035004e+00  2.90595473e-01 -8.87454204e+00  2.50265589e+00
     -6.98332216e+00  3.27459360e+00 -1.92484812e+00  8.73636647e-01
      7.22562800e+00  5.13310754e+00  5.49342038e+00 -8.48421440e-01
      2.84548850e+00  4.62957249e+00  9.38481391e-01 -4.01398541e+00
     -5.58492627e+00  9.42428903e-01  2.57144815e+00 -4.11660947e+00
      4.11475924e+00  4.20385057e+00  1.11384521e+00 -9.04539401e+00
      2.67046541e-02  6.83091903e+00  1.94737781e-01  3.24358065e+00
     -8.75387343e+00  3.84071888e+00  1.00727365e+00  3.90218044e+00
      2.93514417e+00 -4.07601131e+00  2.99660573e+00  3.34676904e+00
      1.72451615e+00  6.66965227e+00  3.40146424e+00  2.17335600e+00
     -2.32970663e+00 -2.23384894e+00  5.68513576e+00  3.95067362e+00
      5.18836707e+00 -1.32827423e+00 -3.47548991e+00 -3.85610183e+00
      7.35926500e+00  3.01746914e+00 -2.60036381e+00  2.51675294e+00
     -2.55920132e+00 -9.29236893e+00  4.03920062e+00  4.37188370e+00
     -1.96995812e+00  6.57110342e-01  2.58836404e+00  1.21646926e+00
      3.21989840e+00  3.69298483e+00 -2.35057004e+00  2.77387740e+00
      1.99517333e+00  2.98645619e+00 -5.91760935e+00  1.77582818e+00
      7.75340964e+00 -7.61147333e+00 -8.09865682e+00  3.44939309e+00
     -8.72342481e+00 -6.95343788e+00 -6.29510112e-02 -5.16258753e+00
      9.24948673e-01  1.75891228e+00  5.62367420e+00  3.22666476e+00
      2.48912317e+00  6.66965227e+00 -6.25649445e+00 -6.93921032e-01
      6.29242353e+00 -5.14623597e+00  8.77584160e-01  5.06487961e+00
      5.48327084e+00  3.41161378e+00  1.43299557e+00 -8.85367863e+00
      5.46240743e+00 -5.61932240e+00  3.40146424e+00 -2.00435425e+00
     -8.14025313e-01  5.38008246e+00  4.07359675e+00  1.11046203e+00
     -5.96215501e+00 -6.01346704e+00  4.79027492e+00  3.90894680e+00
     -6.76454134e+00  5.44210835e+00  3.61404303e+00  1.09029628e-01
      3.23343111e+00  4.59517636e+00  4.28617554e+00  4.27602600e+00
     -3.82508888e+00  4.95154169e+00 -9.36948441e-01  1.85138679e+00
      4.37526687e+00  3.87116749e+00  2.01941992e+00  4.89008013e+00
      5.40094587e+00 -3.41684719e+00  7.07789396e+00 -2.98154006e+00
      2.00870605e+00  4.68370336e+00 -9.26798902e-01 -1.63050869e+00
      3.87455067e+00 -8.22270861e+00  3.51480216e+00  3.78207616e+00
     -6.31570150e+00  1.50460667e+00 -2.65562335e+00 -1.22170267e+00
      2.85281919e+00  6.21009855e+00  4.82805423e+00  3.03100186e+00
      1.42284603e+00 -3.78787390e+00 -5.07744371e+00 -2.49153773e+00]
    ---------------------------------------------------------------------
    fpr:  [0.         0.         0.         0.00518135 0.00518135 0.01036269
     0.01036269 0.01036269 0.01036269 0.01554404 0.01554404 0.02072539
     0.02072539 0.02590674 0.02590674 0.02590674 0.02590674 0.03108808
     0.03108808 0.03108808 0.03108808 0.03108808 0.03108808 0.03626943
     0.03626943 0.03626943 0.03626943 0.03626943 0.03626943 0.03626943
     0.03626943 0.04145078 0.04145078 0.04145078 0.04145078 0.04663212
     0.04663212 0.05181347 0.05181347 0.05181347 0.05181347 0.05181347
     0.05181347 0.05181347 0.05699482 0.05699482 0.05699482 0.05699482
     0.05699482 0.05699482 0.05699482 0.05699482 0.06217617 0.06217617
     0.06735751 0.06735751 0.07253886 0.07253886 0.09326425 0.09326425
     0.10362694 0.10362694 0.10880829 0.10880829 0.10880829 0.11917098
     0.11917098 0.12435233 0.12435233 0.12953368 0.12953368 0.13471503
     0.13471503 0.14507772 0.14507772 0.16062176 0.16062176 0.22279793
     0.22279793 0.23316062 0.37305699 0.37305699 0.38341969 1.        ]
    tpr:  [0.         0.00325733 0.05537459 0.05863192 0.16938111 0.16938111
     0.29641694 0.31596091 0.36156352 0.36156352 0.41042345 0.41042345
     0.42019544 0.42019544 0.43648208 0.44299674 0.46905537 0.46905537
     0.50814332 0.51465798 0.5732899  0.57980456 0.58306189 0.58306189
     0.58631922 0.60586319 0.61237785 0.61889251 0.63517915 0.64169381
     0.64495114 0.64495114 0.64820847 0.65472313 0.66123779 0.66123779
     0.68403909 0.68403909 0.72312704 0.72964169 0.74267101 0.74918567
     0.752443   0.75895765 0.75895765 0.78175896 0.78827362 0.7980456
     0.80456026 0.83061889 0.83713355 0.84039088 0.84039088 0.88925081
     0.88925081 0.90228013 0.90228013 0.90879479 0.90879479 0.9218241
     0.9218241  0.94136808 0.94136808 0.94788274 0.95765472 0.95765472
     0.96091205 0.96091205 0.96742671 0.96742671 0.97394137 0.97394137
     0.98371336 0.98371336 0.98697068 0.98697068 0.99348534 0.99348534
     0.99674267 0.99674267 0.99674267 1.         1.         1.        ]
    thresholds:  [            inf  8.89580973e+00  6.71081475e+00  6.66965227e+00
      5.44210835e+00  5.40094587e+00  4.37865005e+00  4.36850052e+00
      4.11475924e+00  4.09784334e+00  3.90218044e+00  3.88188137e+00
      3.84748524e+00  3.84071888e+00  3.73414732e+00  3.69975119e+00
      3.61404303e+00  3.55596465e+00  3.40484742e+00  3.40146424e+00
      3.04791776e+00  3.03776822e+00  3.03100186e+00  3.01746914e+00
      3.00393642e+00  2.98645619e+00  2.93514417e+00  2.85620237e+00
      2.78402694e+00  2.77387740e+00  2.74963081e+00  2.69775445e+00
      2.65377312e+00  2.64024040e+00  2.62952653e+00  2.61937699e+00
      2.50265589e+00  2.49588953e+00  2.01941992e+00  2.00870605e+00
      1.93371177e+00  1.93032859e+00  1.86153633e+00  1.85138679e+00
      1.80007477e+00  1.72451615e+00  1.70365274e+00  1.67263979e+00
      1.62471095e+00  1.46006100e+00  1.45667782e+00  1.43299557e+00
      1.43243123e+00  9.24948673e-01  9.21565493e-01  8.22324622e-01
      8.12175083e-01  7.15753059e-01  5.20090164e-01  4.45095881e-01
      3.73484779e-01  1.94737781e-01  1.49627782e-01  1.09029628e-01
      5.84124443e-03 -6.29510112e-02 -9.00164473e-02 -1.27795755e-01
     -1.96023678e-01 -2.33802985e-01 -2.61432754e-01 -2.86243675e-01
     -3.71387496e-01 -4.57095649e-01 -5.42803802e-01 -6.93921032e-01
     -7.89214391e-01 -1.22170267e+00 -1.24256608e+00 -1.32827423e+00
     -2.55920132e+00 -2.60036381e+00 -2.65562335e+00 -1.06332508e+01]



    
![png](output_36_1.png)
    


# 3. Logistic Regression for Multi-Class Classification: Iris flower data

## 3.1 Read data and Quick data check


```python
df = pd.read_csv(input_dir + 'iris.csv')
```


```python
df.head()
```





  <div id="df-3f7c1dc4-e280-45f0-a7c2-7d1a371ef2de" class="colab-df-container">
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3f7c1dc4-e280-45f0-a7c2-7d1a371ef2de')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
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

    .colab-df-buttons div {
      margin-bottom: 4px;
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
        document.querySelector('#df-3f7c1dc4-e280-45f0-a7c2-7d1a371ef2de button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3f7c1dc4-e280-45f0-a7c2-7d1a371ef2de');
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


<div id="df-4b699b90-cc96-496c-b5be-d2f90e856688">
  <button class="colab-df-quickchart" onclick="quickchart('df-4b699b90-cc96-496c-b5be-d2f90e856688')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4b699b90-cc96-496c-b5be-d2f90e856688 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   sepal_length  150 non-null    float64
     1   sepal_width   150 non-null    float64
     2   petal_length  150 non-null    float64
     3   petal_width   150 non-null    float64
     4   species       150 non-null    object 
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB



```python
df.describe()
```





  <div id="df-9f9f103b-e7e9-4dcd-9e13-d0a19ea16bc7" class="colab-df-container">
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.054000</td>
      <td>3.758667</td>
      <td>1.198667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.433594</td>
      <td>1.764420</td>
      <td>0.763161</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9f9f103b-e7e9-4dcd-9e13-d0a19ea16bc7')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
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

    .colab-df-buttons div {
      margin-bottom: 4px;
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
        document.querySelector('#df-9f9f103b-e7e9-4dcd-9e13-d0a19ea16bc7 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9f9f103b-e7e9-4dcd-9e13-d0a19ea16bc7');
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


<div id="df-e99d8d2b-bed5-424c-858e-6cdd26d3955e">
  <button class="colab-df-quickchart" onclick="quickchart('df-e99d8d2b-bed5-424c-858e-6cdd26d3955e')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e99d8d2b-bed5-424c-858e-6cdd26d3955e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df['species'].value_counts()
```




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
      <th>count</th>
    </tr>
    <tr>
      <th>species</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>setosa</th>
      <td>50</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>50</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



## 3.2 Exploratory Data Analysis (EDA)


```python
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species');
```


    
![png](output_45_0.png)
    



```python
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species');
```


    
![png](output_46_0.png)
    



```python
sns.pairplot(df, hue='species');
```


    
![png](output_47_0.png)
    



```python
sns.heatmap(df.corr(numeric_only=True), annot=True);
```


    
![png](output_48_0.png)
    



```python
c = df['species'].map({'setosa':0, 'versicolor':1, 'virginica':2})

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['sepal_width'],df['petal_width'],df['sepal_length'],c=c);
```


    
![png](output_49_0.png)
    


## 3.3 Modeling


```python
X = df.drop('species', axis=1)
y = df['species']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)
```


```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegressionCV

model = OneVsRestClassifier(LogisticRegressionCV(Cs=10, penalty='l2', solver='saga'))
#model = OneVsRestClassifier(LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear'))
model.fit(scaled_X_train,y_train)

y_pred = model.predict(scaled_X_test)
y_pred_prob = model.predict_proba(scaled_X_test)
```


```python
y_pred = model.predict(scaled_X_test)
y_pred_prob = model.predict_proba(scaled_X_test)
```


```python
X_train.iloc[0]
```




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
      <th>13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sepal_length</th>
      <td>4.3</td>
    </tr>
    <tr>
      <th>sepal_width</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>petal_length</th>
      <td>1.1</td>
    </tr>
    <tr>
      <th>petal_width</th>
      <td>0.1</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>




```python
scaled_X_train[0]
```




    array([-1.8355105 , -0.18952694, -1.46595274, -1.37800018])




```python
y_train.iloc[0]
```




    'setosa'




```python
model.predict_proba(scaled_X_train[0].reshape(1, -1))
```




    array([[7.65623894e-01, 2.34376106e-01, 1.53525686e-14]])




```python
model.predict(scaled_X_train[0].reshape(1, -1))
```




    array(['setosa'], dtype='<U10')



## 3.4 Evaluating: Confusion Matrix and Receiver Operating Characteristic (ROC) curve for Multi-Class Classification


```python
report_evaluation('Multi-Class Logistic Regression', y_test, y_pred)
```

                  precision    recall  f1-score   support
    
          setosa       1.00      0.92      0.96        13
      versicolor       0.95      0.95      0.95        20
       virginica       0.92      1.00      0.96        12
    
        accuracy                           0.96        45
       macro avg       0.96      0.96      0.96        45
    weighted avg       0.96      0.96      0.96        45
    



    
![png](output_61_1.png)
    



```python
from sklearn.metrics import roc_curve, auc

def plot_ind_multiclass_roc(model, scaled_X_test, y_test, n_classes):

    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    y_score = model.decision_function(scaled_X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, axes = plt.subplots(1, n_classes, figsize=(4*n_classes, 4))
    for i in range(n_classes):
        axes[i].plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        axes[i].plot([0, 1], [0, 1], 'k--')
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title('ROC Curve for Class {}'.format(i))
        axes[i].legend(loc="lower right")
    plt.show();
```


```python
plot_ind_multiclass_roc(model, scaled_X_test, y_test, 3)
```


    
![png](output_63_0.png)
    



```python
from sklearn.metrics import roc_curve, auc

def plot_one_multiclass_roc(model, scaled_X_test, y_test, n_classes):

    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    y_score = model.decision_function(scaled_X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig = plt.figure(figsize=(6, 6))
    axes = fig.add_subplot(111)
    for i in range(n_classes):
        axes.plot(fpr[i], tpr[i], label='Class %d AUC = %0.2f' % (i, roc_auc[i]))
    axes.plot([0, 1], [0, 1], 'k--')
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])
    axes.set_xlabel('False Positive Rate')
    axes.set_ylabel('True Positive Rate')
    axes.set_title('ROC Curve for Multi-Class Classification')
    axes.legend(loc="lower right")
    plt.show();
```


```python
plot_one_multiclass_roc(model, scaled_X_test, y_test, 3)
```


    
![png](output_65_0.png)
    

