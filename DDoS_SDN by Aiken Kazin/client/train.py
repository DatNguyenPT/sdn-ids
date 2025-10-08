# import numpy as np
# from tensorflow.keras import layers, models, callbacks
# from sklearn.preprocessing import StandardScaler

# import pandas as pd
# import time

# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
# import seaborn as sns

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import (
#     RandomForestClassifier, 
#     IsolationForest
# )

# from sklearn import (
#     metrics,
#     preprocessing,
#     metrics,
#     svm
# )

# from sklearn.metrics import (
#     precision_recall_curve,
#     roc_curve,
#     auc,
#     classification_report,
#     roc_auc_score,
#     confusion_matrix,
#     accuracy_score,
#     precision_score, 
#     recall_score, 
#     f1_score
# )

# from sklearn.model_selection import (
#     train_test_split,
#     cross_val_score,
#     cross_val_predict,
#     GridSearchCV
# )

# from sklearn.preprocessing import (
#     StandardScaler, 
#     MinMaxScaler
# )

# from sklearn.feature_selection import (
#     SelectKBest, 
#     f_classif
# )
# from imblearn.over_sampling import SMOTE

# import tensorflow as tf
# from tensorflow.keras import (
#     layers, 
#     models, 
#     callbacks
# )

# from xgboost import XGBClassifier

# class NNModels:
#     def __init__(self, X, y):
#         self.data = X
#         self.labels = y
#         self.results = {}

#         X_scaled = StandardScaler().fit_transform(X)
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             X_scaled, y, random_state=42, test_size=0.3, stratify=y
#         )


#     def MLP(self, epochs=50, batch_size=64):
#         print("\n=== MLP (Dense) ===")
#         start_time = time.time()

#         input_dim = self.X_train.shape[1]
#         model = models.Sequential([
#             layers.Input(shape=(input_dim,)),
#             layers.Dense(128, activation='relu'),
#             layers.Dropout(0.3),
#             layers.Dense(64, activation='relu'),
#             layers.Dropout(0.2),
#             layers.Dense(32, activation='relu'),
#             layers.Dense(1, activation='sigmoid')
#         ])
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#         es = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
#         history = model.fit(self.X_train, self.y_train, validation_split=0.1,
#                             epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)

#         y_proba = model.predict(self.X_test).ravel()
#         y_pred = (y_proba > 0.5).astype(int)

#         print(classification_report(self.y_test, y_pred))
#         self.results["MLP"] = {
#             "accuracy": accuracy_score(self.y_test, y_pred),
#             "precision": precision_score(self.y_test, y_pred, zero_division=0),
#             "recall": recall_score(self.y_test, y_pred, zero_division=0),
#             "f1": f1_score(self.y_test, y_pred, zero_division=0)
#         }
#         print("--- %s seconds ---" % (time.time() - start_time))
#         return model, history

#     def GRU(self, epochs=50, batch_size=64, timesteps=1):
#         """
#         Similar to LSTM but using GRU layers.
#         """
#         print("\n=== GRU ===")
#         start_time = time.time()

#         n_features = self.X_train.shape[1]
#         X_train_seq = self.X_train.reshape((self.X_train.shape[0], timesteps, n_features))
#         X_test_seq = self.X_test.reshape((self.X_test.shape[0], timesteps, n_features))

#         model = models.Sequential([
#             layers.Input(shape=(timesteps, n_features)),
#             layers.GRU(64, return_sequences=False),
#             layers.Dropout(0.3),
#             layers.Dense(32, activation='relu'),
#             layers.Dense(1, activation='sigmoid')
#         ])
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#         es = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
#         history = model.fit(X_train_seq, self.y_train, validation_split=0.1, epochs=epochs,
#                             batch_size=batch_size, callbacks=[es], verbose=0)

#         y_proba = model.predict(X_test_seq).ravel()
#         y_pred = (y_proba > 0.5).astype(int)

#         print(classification_report(self.y_test, y_pred))
#         self.results["GRU"] = {
#             "accuracy": accuracy_score(self.y_test, y_pred),
#             "precision": precision_score(self.y_test, y_pred, zero_division=0),
#             "recall": recall_score(self.y_test, y_pred, zero_division=0),
#             "f1": f1_score(self.y_test, y_pred, zero_division=0)
#         }
#         print("--- %s seconds ---" % (time.time() - start_time))
#         return model, history

#     def Conv1D(self, epochs=50, batch_size=64):
#         """
#         1D-CNN for tabular data: we reshape to (samples, features, 1) and apply Conv1D.
#         """
#         print("\n=== 1D-CNN ===")
#         start_time = time.time()

#         n_features = self.X_train.shape[1]
#         # reshape to (samples, timesteps=n_features, channels=1)
#         X_train_cnn = self.X_train.reshape((self.X_train.shape[0], n_features, 1))
#         X_test_cnn = self.X_test.reshape((self.X_test.shape[0], n_features, 1))

#         model = models.Sequential([
#             layers.Input(shape=(n_features, 1)),
#             layers.Conv1D(64, kernel_size=3, activation='relu'),
#             layers.MaxPooling1D(pool_size=2),
#             layers.Conv1D(32, kernel_size=3, activation='relu'),
#             layers.GlobalMaxPooling1D(),
#             layers.Dense(32, activation='relu'),
#             layers.Dense(1, activation='sigmoid')
#         ])
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#         es = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
#         history = model.fit(X_train_cnn, self.y_train, validation_split=0.1,
#                             epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)

#         y_proba = model.predict(X_test_cnn).ravel()
#         y_pred = (y_proba > 0.5).astype(int)

#         print(classification_report(self.y_test, y_pred))
#         self.results["1D-CNN"] = {
#             "accuracy": accuracy_score(self.y_test, y_pred),
#             "precision": precision_score(self.y_test, y_pred, zero_division=0),
#             "recall": recall_score(self.y_test, y_pred, zero_division=0),
#             "f1": f1_score(self.y_test, y_pred, zero_division=0)
#         }
#         print("--- %s seconds ---" % (time.time() - start_time))
#         return model, history

import numpy as np
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NNModels:
    def __init__(self, X, y):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(X)
        self.y = y.values if hasattr(y, "values") else y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

    def MLP(self):
        model = models.Sequential([
            layers.Input(shape=(self.X_train.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
