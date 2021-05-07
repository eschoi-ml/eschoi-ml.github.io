[<-PREV](enronemail.md)

# The Enron Email Dataset

# Part 1. 
1. Import libraries, set up directories, and read data
2. Quick data check
3. Feature engineering
4. Exploratory data analysis

## 1. Import libraries, set up directories, and read data


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import email 
```


```python
input_dir = '../input/enron-email-dataset/'
output_dir = ''
```


```python
df = pd.read_csv(input_dir + 'emails.csv')
```

## 2. Quick data check


```python
print(df.info())
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 517401 entries, 0 to 517400
    Data columns (total 2 columns):
     #   Column   Non-Null Count   Dtype 
    ---  ------   --------------   ----- 
     0   file     517401 non-null  object
     1   message  517401 non-null  object
    dtypes: object(2)
    memory usage: 7.9+ MB
    None





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
      <th>file</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>allen-p/_sent_mail/1.</td>
      <td>Message-ID: &lt;18782981.1075855378110.JavaMail.e...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>allen-p/_sent_mail/10.</td>
      <td>Message-ID: &lt;15464986.1075855378456.JavaMail.e...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>allen-p/_sent_mail/100.</td>
      <td>Message-ID: &lt;24216240.1075855687451.JavaMail.e...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>allen-p/_sent_mail/1000.</td>
      <td>Message-ID: &lt;13505866.1075863688222.JavaMail.e...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>allen-p/_sent_mail/1001.</td>
      <td>Message-ID: &lt;30922949.1075863688243.JavaMail.e...</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.iloc[1]['file'])
```

    allen-p/_sent_mail/10.



```python
print(df.iloc[1]['message'])
```

    Message-ID: <15464986.1075855378456.JavaMail.evans@thyme>
    Date: Fri, 4 May 2001 13:51:00 -0700 (PDT)
    From: phillip.allen@enron.com
    To: john.lavorato@enron.com
    Subject: Re:
    Mime-Version: 1.0
    Content-Type: text/plain; charset=us-ascii
    Content-Transfer-Encoding: 7bit
    X-From: Phillip K Allen
    X-To: John J Lavorato <John J Lavorato/ENRON@enronXgate@ENRON>
    X-cc: 
    X-bcc: 
    X-Folder: \Phillip_Allen_Jan2002_1\Allen, Phillip K.\'Sent Mail
    X-Origin: Allen-P
    X-FileName: pallen (Non-Privileged).pst
    
    Traveling to have a business meeting takes the fun out of the trip.  Especially if you have to prepare a presentation.  I would suggest holding the business plan meetings here then take a trip without any formal business meetings.  I would even try and get some honest opinions on whether a trip is even desired or necessary.
    
    As far as the business meetings, I think it would be more productive to try and stimulate discussions across the different groups about what is working and what is not.  Too often the presenter speaks and the others are quiet just waiting for their turn.   The meetings might be better if held in a round table discussion format.  
    
    My suggestion for where to go is Austin.  Play golf and rent a ski boat and jet ski's.  Flying somewhere takes too much time.
    


## 3. Feature engineering

### 3.1 Extract data


```python
header_cols = ['Date', 'From', 'To','Subject']
df[header_cols + ['Body']] = None
for i in range(df.shape[0]):
    
    e = email.message_from_string(df.loc[i]['message'])
    for col in header_cols:
        df.loc[i][col] = e.get(col)
    df.loc[i]['Body'] = e.get_payload()
```


```python
def split_email_addresses(x):
    if x:
        addrs = x.split(',')
        addrs = [s.strip() for s in addrs]
    else:
        addrs = None
    return addrs

df['To'] = df['To'].apply(split_email_addresses)
```


```python
def DirectVSBroadcast(x):
    if x is None:
        return None
    elif len(x) == 1:
        return 'direct'
    else:
        return 'broadcast'

df['DvsB'] = df['To'].apply(DirectVSBroadcast)
```


```python
print(df.info())
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 517401 entries, 0 to 517400
    Data columns (total 8 columns):
     #   Column   Non-Null Count   Dtype 
    ---  ------   --------------   ----- 
     0   file     517401 non-null  object
     1   message  517401 non-null  object
     2   Date     517401 non-null  object
     3   From     517401 non-null  object
     4   To       495554 non-null  object
     5   Subject  517401 non-null  object
     6   Body     517401 non-null  object
     7   DvsB     495554 non-null  object
    dtypes: object(8)
    memory usage: 31.6+ MB
    None





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
      <th>file</th>
      <th>message</th>
      <th>Date</th>
      <th>From</th>
      <th>To</th>
      <th>Subject</th>
      <th>Body</th>
      <th>DvsB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>allen-p/_sent_mail/1.</td>
      <td>Message-ID: &lt;18782981.1075855378110.JavaMail.e...</td>
      <td>Mon, 14 May 2001 16:39:00 -0700 (PDT)</td>
      <td>phillip.allen@enron.com</td>
      <td>[tim.belden@enron.com]</td>
      <td></td>
      <td>Here is our forecast\n\n</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>1</th>
      <td>allen-p/_sent_mail/10.</td>
      <td>Message-ID: &lt;15464986.1075855378456.JavaMail.e...</td>
      <td>Fri, 4 May 2001 13:51:00 -0700 (PDT)</td>
      <td>phillip.allen@enron.com</td>
      <td>[john.lavorato@enron.com]</td>
      <td>Re:</td>
      <td>Traveling to have a business meeting takes the...</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>2</th>
      <td>allen-p/_sent_mail/100.</td>
      <td>Message-ID: &lt;24216240.1075855687451.JavaMail.e...</td>
      <td>Wed, 18 Oct 2000 03:00:00 -0700 (PDT)</td>
      <td>phillip.allen@enron.com</td>
      <td>[leah.arsdall@enron.com]</td>
      <td>Re: test</td>
      <td>test successful.  way to go!!!</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>3</th>
      <td>allen-p/_sent_mail/1000.</td>
      <td>Message-ID: &lt;13505866.1075863688222.JavaMail.e...</td>
      <td>Mon, 23 Oct 2000 06:13:00 -0700 (PDT)</td>
      <td>phillip.allen@enron.com</td>
      <td>[randall.gay@enron.com]</td>
      <td></td>
      <td>Randy,\n\n Can you send me a schedule of the s...</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>4</th>
      <td>allen-p/_sent_mail/1001.</td>
      <td>Message-ID: &lt;30922949.1075863688243.JavaMail.e...</td>
      <td>Thu, 31 Aug 2000 05:07:00 -0700 (PDT)</td>
      <td>phillip.allen@enron.com</td>
      <td>[greg.piper@enron.com]</td>
      <td>Re: Hello</td>
      <td>Let's shoot for Tuesday at 11:45.</td>
      <td>direct</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Exploratory data analysis 

### 4.1 Sender analysis
1. Top 10 senders who sent the most number of emails
2. Top 10 senders who sent the most number of direct emails
3. Top 10 senders who sent the most number of broadcast emails
4. Top 20 most number of emails and its sender per sending


```python
df['From'].value_counts()[:10]
```




    kay.mann@enron.com               16735
    vince.kaminski@enron.com         14368
    jeff.dasovich@enron.com          11411
    pete.davis@enron.com              9149
    chris.germany@enron.com           8801
    sara.shackleton@enron.com         8777
    enron.announcements@enron.com     8587
    tana.jones@enron.com              8490
    steven.kean@enron.com             6759
    kate.symes@enron.com              5438
    Name: From, dtype: int64




```python
df[df['DvsB']=='direct']['From'].value_counts()[:10]
```




    vince.kaminski@enron.com         13742
    kay.mann@enron.com               13693
    pete.davis@enron.com              9148
    enron.announcements@enron.com     8406
    sara.shackleton@enron.com         7274
    jeff.dasovich@enron.com           7140
    tana.jones@enron.com              5768
    chris.germany@enron.com           5637
    steven.kean@enron.com             5387
    kate.symes@enron.com              4935
    Name: From, dtype: int64




```python
df[df['DvsB']=='broadcast']['From'].value_counts()[:10]
```




    jeff.dasovich@enron.com      4247
    chris.germany@enron.com      3108
    kay.mann@enron.com           3028
    tana.jones@enron.com         2679
    sara.shackleton@enron.com    1483
    susan.mara@enron.com         1349
    mjones7@txu.com              1056
    eric.bass@enron.com           971
    james.steffes@enron.com       913
    david.delainey@enron.com      851
    Name: From, dtype: int64




```python
sorted_count = df['To'].apply(lambda x: len(x) if x else 0).sort_values(ascending=False)[:20]
sorted_count = pd.DataFrame(sorted_count)
sorted_count.rename(columns = {'To':'Count'}, inplace=True)
sorted_count['From'] = df.loc[sorted_count.index]['From']
sorted_count
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
      <th>Count</th>
      <th>From</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>390004</th>
      <td>1029</td>
      <td>lisa.jones@enron.com</td>
    </tr>
    <tr>
      <th>39106</th>
      <td>946</td>
      <td>kenneth.lay@enron.com</td>
    </tr>
    <tr>
      <th>19062</th>
      <td>946</td>
      <td>kenneth.lay@enron.com</td>
    </tr>
    <tr>
      <th>19856</th>
      <td>946</td>
      <td>kenneth.lay@enron.com</td>
    </tr>
    <tr>
      <th>333576</th>
      <td>946</td>
      <td>kenneth.lay@enron.com</td>
    </tr>
    <tr>
      <th>335058</th>
      <td>946</td>
      <td>kenneth.lay@enron.com</td>
    </tr>
    <tr>
      <th>380164</th>
      <td>946</td>
      <td>kenneth.lay@enron.com</td>
    </tr>
    <tr>
      <th>497284</th>
      <td>945</td>
      <td>kenneth.lay@enron.com</td>
    </tr>
    <tr>
      <th>273473</th>
      <td>945</td>
      <td>kenneth.lay@enron.com</td>
    </tr>
    <tr>
      <th>392169</th>
      <td>933</td>
      <td>constance.charles@enron.com</td>
    </tr>
    <tr>
      <th>497344</th>
      <td>933</td>
      <td>constance.charles@enron.com</td>
    </tr>
    <tr>
      <th>275916</th>
      <td>933</td>
      <td>constance.charles@enron.com</td>
    </tr>
    <tr>
      <th>512539</th>
      <td>890</td>
      <td>tracey.kozadinos@enron.com</td>
    </tr>
    <tr>
      <th>170372</th>
      <td>795</td>
      <td>bob.ambrocik@enron.com</td>
    </tr>
    <tr>
      <th>390126</th>
      <td>795</td>
      <td>bob.ambrocik@enron.com</td>
    </tr>
    <tr>
      <th>445807</th>
      <td>795</td>
      <td>bob.ambrocik@enron.com</td>
    </tr>
    <tr>
      <th>445795</th>
      <td>795</td>
      <td>bob.ambrocik@enron.com</td>
    </tr>
    <tr>
      <th>390168</th>
      <td>795</td>
      <td>bob.ambrocik@enron.com</td>
    </tr>
    <tr>
      <th>390116</th>
      <td>795</td>
      <td>bob.ambrocik@enron.com</td>
    </tr>
    <tr>
      <th>360299</th>
      <td>795</td>
      <td>bob.ambrocik@enron.com</td>
    </tr>
  </tbody>
</table>
</div>



### 4.2 Receipient analysis
1. Top 10 receipients who received the most number of direct emails
2. Top 10 receipients who received the most number of broadcast emails


```python
from collections import defaultdict

direct_receip = defaultdict(int)
broadcast_receip = defaultdict(int)

def count_receipient(x):
    if x:        
        if len(x)==1:
            direct_receip[x[0]] += 1
        else:
            for receip in x:
                broadcast_receip[receip] += 1

df['To'].apply(count_receipient)
```




    0         None
    1         None
    2         None
    3         None
    4         None
              ... 
    517396    None
    517397    None
    517398    None
    517399    None
    517400    None
    Name: To, Length: 517401, dtype: object




```python
direct_receip = pd.Series(direct_receip)
broadcast_receip = pd.Series(broadcast_receip)
```


```python
direct_receip.sort_values(ascending=False)[:10]
```




    pete.davis@enron.com         9155
    tana.jones@enron.com         5677
    sara.shackleton@enron.com    4974
    vkaminski@aol.com            4870
    jeff.dasovich@enron.com      4350
    kate.symes@enron.com         3517
    all.worldwide@enron.com      3324
    mark.taylor@enron.com        3295
    kay.mann@enron.com           3085
    gerald.nemec@enron.com       3074
    dtype: int64




```python
broadcast_receip.sort_values(ascending=False)[:10]
```




    richard.shapiro@enron.com    13122
    jeff.dasovich@enron.com       9857
    steven.kean@enron.com         9724
    james.steffes@enron.com       9646
    susan.mara@enron.com          8636
    paul.kaufman@enron.com        8179
    tim.belden@enron.com          7594
    tana.jones@enron.com          7151
    mark.taylor@enron.com         6492
    sara.shackleton@enron.com     6459
    dtype: int64



### 4.3 Sender-Receipient pair analysis 
- Top 10 sender-receipient pairs 


```python
sub_df = df[['From', 'To', 'DvsB']]
sub_df = sub_df[sub_df['DvsB']=='direct']
sub_df['To'] = sub_df['To'].apply(lambda x: str(x[0]))
```


```python
sub_df
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
      <th>From</th>
      <th>To</th>
      <th>DvsB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>phillip.allen@enron.com</td>
      <td>tim.belden@enron.com</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>1</th>
      <td>phillip.allen@enron.com</td>
      <td>john.lavorato@enron.com</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>2</th>
      <td>phillip.allen@enron.com</td>
      <td>leah.arsdall@enron.com</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>3</th>
      <td>phillip.allen@enron.com</td>
      <td>randall.gay@enron.com</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>4</th>
      <td>phillip.allen@enron.com</td>
      <td>greg.piper@enron.com</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>517396</th>
      <td>john.zufferli@enron.com</td>
      <td>kori.loibl@enron.com</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>517397</th>
      <td>john.zufferli@enron.com</td>
      <td>john.lavorato@enron.com</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>517398</th>
      <td>john.zufferli@enron.com</td>
      <td>dawn.doucet@enron.com</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>517399</th>
      <td>john.zufferli@enron.com</td>
      <td>jeanie.slone@enron.com</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>517400</th>
      <td>john.zufferli@enron.com</td>
      <td>livia_zufferli@monitor.com</td>
      <td>direct</td>
    </tr>
  </tbody>
</table>
<p>354298 rows × 3 columns</p>
</div>




```python
sub_df = sub_df.groupby(['From', 'To']).count().reset_index()
sub_df.rename(columns = {'DvsB':'Count'}, inplace = True)
```


```python
sub_df
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
      <th>From</th>
      <th>To</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>'todd'.delahoussaye@enron.com</td>
      <td>susan.bailey@enron.com</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>--migrated--bmishkin@ercot.com</td>
      <td>mockmarket@ercot.com</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>--migrated--dodle@ercot.com</td>
      <td>set@ercot.com</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-nikole@excite.com</td>
      <td>bill.williams@enron.com</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-persson@ricemail.ricefinancial.com</td>
      <td>barry.tycholiz@enron.com</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>56114</th>
      <td>zufferli@enron.com</td>
      <td>john.lavorato@enron.com</td>
      <td>2</td>
    </tr>
    <tr>
      <th>56115</th>
      <td>zvo2z17d0@untappedmarkets.com</td>
      <td>undisclosed-recipients@enron.com</td>
      <td>1</td>
    </tr>
    <tr>
      <th>56116</th>
      <td>zwharton@dawray.com</td>
      <td>martin.cuilla@enron.com</td>
      <td>12</td>
    </tr>
    <tr>
      <th>56117</th>
      <td>zwharton@dawray.com</td>
      <td>mcuilla@enron.com</td>
      <td>15</td>
    </tr>
    <tr>
      <th>56118</th>
      <td>zzmacmac@aol.com</td>
      <td>j.kaminski@enron.com</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>56119 rows × 3 columns</p>
</div>




```python
sub_df.sort_values(by=['Count'], ascending=False)[:10]
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
      <th>From</th>
      <th>To</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40677</th>
      <td>pete.davis@enron.com</td>
      <td>pete.davis@enron.com</td>
      <td>9141</td>
    </tr>
    <tr>
      <th>54820</th>
      <td>vince.kaminski@enron.com</td>
      <td>vkaminski@aol.com</td>
      <td>4308</td>
    </tr>
    <tr>
      <th>14602</th>
      <td>enron.announcements@enron.com</td>
      <td>all.worldwide@enron.com</td>
      <td>2206</td>
    </tr>
    <tr>
      <th>14595</th>
      <td>enron.announcements@enron.com</td>
      <td>all.houston@enron.com</td>
      <td>1701</td>
    </tr>
    <tr>
      <th>27515</th>
      <td>kay.mann@enron.com</td>
      <td>suzanne.adams@enron.com</td>
      <td>1528</td>
    </tr>
    <tr>
      <th>54750</th>
      <td>vince.kaminski@enron.com</td>
      <td>shirley.crenshaw@enron.com</td>
      <td>1190</td>
    </tr>
    <tr>
      <th>49985</th>
      <td>steven.kean@enron.com</td>
      <td>maureen.mcvicker@enron.com</td>
      <td>1014</td>
    </tr>
    <tr>
      <th>27424</th>
      <td>kay.mann@enron.com</td>
      <td>nmann@erac.com</td>
      <td>980</td>
    </tr>
    <tr>
      <th>26845</th>
      <td>kate.symes@enron.com</td>
      <td>evelyn.metoyer@enron.com</td>
      <td>915</td>
    </tr>
    <tr>
      <th>26880</th>
      <td>kate.symes@enron.com</td>
      <td>kerri.thompson@enron.com</td>
      <td>859</td>
    </tr>
  </tbody>
</table>
</div>

[<-PREV](enronemail.md)
