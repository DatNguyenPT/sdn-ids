import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE


def process_col(df):
    drop_cols = ['dt', 'src', 'dst', 'switch']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    obj_cols = df.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].fillna(df[obj_cols].mode().iloc[0])

    if 'dur' in df.columns and 'dur_nsec' in df.columns:
        df['duration_sec'] = df['dur'] + df['dur_nsec'] / 1e9
    elif 'dur' in df.columns:
        df['duration_sec'] = df['dur']
    else:
        df['duration_sec'] = 0

    num_cols = [
        'pktcount', 'bytecount', 'duration_sec', 'flows', 'packetins',
        'pktperflow', 'byteperflow', 'pktrate', 'tx_bytes', 'rx_bytes',
        'tx_kbps', 'rx_kbps', 'tot_kbps', 'port_no'
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


def normalization(df):
    # Convert numeric types
    df['duration_sec'] = pd.to_numeric(df.get('duration_sec', 0), errors='coerce').fillna(0)
    if 'port_no' in df.columns:
        df['port_no'] = pd.to_numeric(df['port_no'], errors='coerce')

    # Remove invalids
    for col in ['pktcount', 'bytecount', 'dur']:
        if col in df.columns:
            df = df[df[col] >= 0]

    # Encode categorical
    if 'protocol' in df.columns:
        le = LabelEncoder()
        df['protocol'] = le.fit_transform(df['protocol'])
        print("Protocol mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # Feature engineering
    df['pkt_per_sec'] = df['pktcount'] / (df['dur'] + 1e-5)
    df['byte_per_pkt'] = df['bytecount'] / (df['pktcount'] + 1e-5)
    df['rx_tx_ratio'] = (df['rx_bytes'] + 1) / (df['tx_bytes'] + 1)
    df['byte_rate'] = df['bytecount'] / (df['dur'] + 1e-5)

    # Log-scale skewed
    for col in ['pktcount', 'bytecount', 'tx_bytes', 'rx_bytes', 'tot_kbps']:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Outlier capping
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = np.clip(df[col], df[col].quantile(0.01), df[col].quantile(0.99))

    df = df.dropna().reset_index(drop=True)
    return df


def split_dataset(df):
    X = df.drop(columns=['label'], errors='ignore')
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def process_data(X_train, X_test, y_train, y_test):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    selector = SelectKBest(f_classif, k=min(10, X_train_res.shape[1]))
    X_train_selected = selector.fit_transform(X_train_res, y_train_res)
    X_test_selected = selector.transform(X_test)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    return X_train_scaled, X_test_scaled, y_train_res, y_test
