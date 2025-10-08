import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

data = pd.read_csv('dataset_sdn.csv')

#### Let's support which columns NUMERIC and which is OBJECT
df = data.copy()
df.columns = df.columns.str.strip().str.lower()
numeric_df = df.select_dtypes(include=['int64', 'float64'])
object_df = df.select_dtypes(include=['object'])
numeric_cols = numeric_df.columns
object_cols = object_df.columns
print('Numeric Columns: ')
print(numeric_cols, '\n')
print('Object Columns: ')
print(object_cols, '\n')
print('Number of Numeric Features: ', len(numeric_cols))
print('Number of Object Features: ', len(object_cols))

def process_col(df):
    # Fill numeric with median
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill object (string/categorical) with mode 
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].fillna(df[obj_cols].mode().iloc[0])
    set(num_cols) - set(df.columns)

    df['duration_sec'] = df['dur'] + df['dur_nsec'] / 1e9
    # Convert to numeric
    num_cols = ['pktcount', 'bytecount', 'duration_sec', 'flows', 'packetins', 
                'pktperflow', 'byteperflow', 'pktrate', 'tx_bytes', 'rx_bytes', 
                'tx_kbps', 'rx_kbps', 'tot_kbps', 'port_no']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    # Fill missing values
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

def normalization(df):
    df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce')
    df['port_no'] = df['port_no'].astype(int)
    df.dtypes

    # Invalid values
    df = df[df['pktcount'] >= 0]

    # remove outliers [-3; 3]
    from scipy import stats
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)] 

    # Label encoder (categorial -> numeric)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['protocol'] = le.fit_transform(df['protocol'])
    protocol_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    print(protocol_mapping)

    # feature engineering
    df['pkt_per_sec'] = df['pktcount'] / (df['dur'] + 1e-5)
    df['byte_per_pkt'] = df['bytecount'] / (df['pktcount'] + 1)

    df.head()
    return df

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

def process_data(df, numeric_cols):
    X = df[numeric_cols]
    y = df['label']
    
    # 1. Split with stratify to keep class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Apply SMOTE on training data only
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 3. Feature selection on resampled train data
    selector = SelectKBest(f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train_res, y_train_res)
    X_test_selected = selector.transform(X_test)  # use same features
    
    # 4. Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    return X_train_scaled, X_test_scaled, y_train_res, y_test
