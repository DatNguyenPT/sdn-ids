# import os
# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")  # Prevent GUI pop-up
# import matplotlib.pyplot as plt
# import tensorflow as tf

# from tensorflow.keras import layers, models
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

# from preprocess import (
#     process_col,
#     normalization,
#     split_dataset
# )


# class NNModels:
#     def __init__(self, df):
#         self.X_train, self.X_test, self.y_train, self.y_test = split_dataset(df)

#     def MLPv2(self):
#         y_train_cat = tf.keras.utils.to_categorical(self.y_train)
#         y_test_cat = tf.keras.utils.to_categorical(self.y_test)

#         model = Sequential([
#             tf.keras.Input(shape=(self.X_train.shape[1],)),
#             Dense(64, activation='relu'),
#             BatchNormalization(),
#             Dropout(0.3),
#             Dense(32, activation='relu'),
#             BatchNormalization(),
#             Dropout(0.3),
#             Dense(y_train_cat.shape[1], activation='softmax')
#         ])

#         model.compile(optimizer=Adam(learning_rate=0.001),
#                       loss='categorical_crossentropy',
#                       metrics=['accuracy'])

#         # === Directories ===
#         os.makedirs("logs", exist_ok=True)
#         os.makedirs("models", exist_ok=True)
#         os.makedirs("img", exist_ok=True)
#         os.makedirs("report", exist_ok=True)

#         # === Callbacks ===
#         tensorboard_cb = TensorBoard(log_dir="logs")
#         early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

#         print("\n🧠 Training MLPv2 model...")
#         history = model.fit(
#             self.X_train, y_train_cat,
#             validation_split=0.2,
#             epochs=50,
#             batch_size=32,
#             callbacks=[tensorboard_cb, early_stop, reduce_lr],
#             verbose=1
#         )

#         # === Save model ===
#         model.save("models/mlp_model.h5")
#         print("✅ Model saved to models/mlp_model.h5")

#         # === Generate plots and report ===
#         self.save_training_report(history)

#         return model

#     @staticmethod
#     def save_training_report(history):
#         # Extract history data
#         hist_df = pd.DataFrame(history.history)
#         hist_df["epoch"] = range(1, len(hist_df) + 1)

#         # === Plot and save ===
#         plt.figure(figsize=(10, 4))

#         # Accuracy
#         plt.subplot(1, 2, 1)
#         plt.plot(hist_df["epoch"], hist_df["accuracy"], label='Train Acc')
#         plt.plot(hist_df["epoch"], hist_df["val_accuracy"], label='Val Acc')
#         plt.title('Accuracy over Epochs')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.legend()

#         # Loss
#         plt.subplot(1, 2, 2)
#         plt.plot(hist_df["epoch"], hist_df["loss"], label='Train Loss')
#         plt.plot(hist_df["epoch"], hist_df["val_loss"], label='Val Loss')
#         plt.title('Loss over Epochs')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()

#         plt.tight_layout()
#         img_path = os.path.join("img", "training_curves.png")
#         plt.savefig(img_path)
#         plt.close()
#         print(f"📈 Training curves saved to {img_path}")

#         # === Save Excel report ===
#         report_path = os.path.join("report", "training_report.xlsx")
#         with pd.ExcelWriter(report_path) as writer:
#             hist_df.to_excel(writer, index=False, sheet_name="Training_History")

#             summary_df = pd.DataFrame({
#                 "Metric": ["Final Train Acc", "Final Val Acc", "Final Train Loss", "Final Val Loss"],
#                 "Value": [
#                     hist_df["accuracy"].iloc[-1],
#                     hist_df["val_accuracy"].iloc[-1],
#                     hist_df["loss"].iloc[-1],
#                     hist_df["val_loss"].iloc[-1]
#                 ]
#             })
#             summary_df.to_excel(writer, index=False, sheet_name="Summary")

#         print(f"📊 Training report saved to {report_path}")


# def main():
#     print("🚀 Loading dataset...")
#     data = pd.read_csv('dataset_sdn.csv')
#     df = data.copy()
#     df.columns = df.columns.str.strip().str.lower()

#     print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns\n")

#     # PROCESS DATA
#     df = process_col(df)
#     df = normalization(df)

#     # TRAIN MODEL
#     model = NNModels(df)
#     model.MLPv2()


# if __name__ == "__main__":
#     main()


import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # No GUI pop-up
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

from preprocess import (
    process_col,
    normalization,
    split_dataset
)


class NNModels:
    def __init__(self, df):
        self.X_train, self.X_test, self.y_train, self.y_test = split_dataset(df)
        self.num_features = self.X_train.shape[1]
        self.num_classes = len(np.unique(self.y_train))

        # Ensure 3D shape for CNN/LSTM
        self.X_train_3d = np.expand_dims(self.X_train, axis=2)
        self.X_test_3d = np.expand_dims(self.X_test, axis=2)

        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("img", exist_ok=True)
        os.makedirs("report", exist_ok=True)

    def _train_and_report(self, model, name):
        """Generic train + report pipeline"""
        y_train_cat = tf.keras.utils.to_categorical(self.y_train)
        y_test_cat = tf.keras.utils.to_categorical(self.y_test)

        log_dir = os.path.join("logs", name)
        os.makedirs(log_dir, exist_ok=True)

        tensorboard_cb = TensorBoard(log_dir=log_dir)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

        print(f"\n Training {name} model...")
        history = model.fit(
            self.X_train_3d if "LSTM" in name or "CNN" in name else self.X_train,
            y_train_cat,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[tensorboard_cb, early_stop, reduce_lr],
            verbose=1
        )

        model_path = os.path.join("models", f"{name}.h5")
        model.save(model_path)
        print(f" Model saved to {model_path}")

        self.save_training_report(history, name)
        return model

    # ---------------- MODELS ---------------- #

    def MLPv2(self):
        model = Sequential([
            tf.keras.Input(shape=(self.num_features,)),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return self._train_and_report(model, "MLPv2")

    def CNN1D(self):
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(self.num_features, 1)),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),

            Conv1D(128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),

            Flatten(),
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return self._train_and_report(model, "CNN1D")

    def LSTMModel(self):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.num_features, 1)),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return self._train_and_report(model, "LSTM")

    def CNN_LSTM(self):
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=(self.num_features, 1)),
            MaxPooling1D(2),
            Dropout(0.3),

            LSTM(64, return_sequences=False),
            Dropout(0.3),

            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return self._train_and_report(model, "CNN_LSTM")

    # ---------------- REPORT ---------------- #

    @staticmethod
    def save_training_report(history, model_name):
        hist_df = pd.DataFrame(history.history)
        hist_df["epoch"] = range(1, len(hist_df) + 1)

        # Plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(hist_df["epoch"], hist_df["accuracy"], label='Train Acc')
        plt.plot(hist_df["epoch"], hist_df["val_accuracy"], label='Val Acc')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(hist_df["epoch"], hist_df["loss"], label='Train Loss')
        plt.plot(hist_df["epoch"], hist_df["val_loss"], label='Val Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        img_path = os.path.join("img", f"{model_name}_curves.png")
        plt.savefig(img_path)
        plt.close()
        print(f" Saved training plot to {img_path}")

        # Excel report
        report_path = os.path.join("report", f"{model_name}_report.xlsx")
        with pd.ExcelWriter(report_path) as writer:
            hist_df.to_excel(writer, index=False, sheet_name="History")
            summary = pd.DataFrame({
                "Metric": ["Final Train Acc", "Final Val Acc", "Final Train Loss", "Final Val Loss"],
                "Value": [
                    hist_df["accuracy"].iloc[-1],
                    hist_df["val_accuracy"].iloc[-1],
                    hist_df["loss"].iloc[-1],
                    hist_df["val_loss"].iloc[-1]
                ]
            })
            summary.to_excel(writer, index=False, sheet_name="Summary")

        print(f" Report saved to {report_path}")


def main():
    print(" Loading dataset...")
    data = pd.read_csv("dataset_sdn.csv")
    df = data.copy()
    df.columns = df.columns.str.strip().str.lower()
    print(f" Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns\n")

    df = process_col(df)
    df = normalization(df)

    trainer = NNModels(df)
    trainer.MLPv2()
    trainer.CNN1D()
    trainer.LSTMModel()
    trainer.CNN_LSTM()


if __name__ == "__main__":
    main()

