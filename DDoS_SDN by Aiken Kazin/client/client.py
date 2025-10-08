import os
import pandas as pd
import flwr as fl
from train import NNModels

# Chọn thuật toán qua env (hiện chỉ MLP)
ALGO = os.getenv("ALGO", "MLP")

# Load dataset riêng cho client
data = pd.read_csv("/app/dataset_sdn.csv")
df = data.dropna()
X = df.drop(['dt','src','dst','label'], axis=1)
y = df['label']
X = pd.get_dummies(X)

nn_models = NNModels(X, y)  

# Chọn model
model = nn_models.MLP()

# Flower client
class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(nn_models.X_train, nn_models.y_train, epochs=3, batch_size=32, verbose=0)
        return model.get_weights(), len(nn_models.X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, acc = model.evaluate(nn_models.X_test, nn_models.y_test, verbose=0)
        return float(loss), len(nn_models.X_test), {"accuracy": float(acc)}

# Start client
fl.client.start_client(
    server_address="supernode:8080",
    client=FLClient().to_client()
)
