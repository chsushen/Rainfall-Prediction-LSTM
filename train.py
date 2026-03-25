import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model import build_model

# dummy rainfall data (replace later)
data = np.random.rand(200, 1)

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

X = []
y = []

seq_len = 10

for i in range(len(data) - seq_len):
    X.append(data[i:i+seq_len])
    y.append(data[i+seq_len])

X = np.array(X)
y = np.array(y)

model = build_model((seq_len, 1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=10)
