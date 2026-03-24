import os
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

districts = ["beed"]#,"Chhatrapati Sambhajinagar","Dhule","Jalgaon","Jalna","Wardha","Yavatmal"]


def train_for_district(district):

    safe_name = district.lower().replace(" ", "_")

    df = pd.read_csv(f"{district}_master_2014_2025.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['month'] = df['date'].dt.month
    train_df = df.copy()
    print("Train period:", train_df['date'].min(), "to", train_df['date'].max())

    feature_cols = [
        'msl', 'wind_speed', 'solar_radiation',
        'relative_humidity', 'rainfall', 'month'
    ]
    target_col = 'tmax'

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_df[feature_cols] = scaler_X.fit_transform(train_df[feature_cols])
    train_df[[target_col]] = scaler_y.fit_transform(train_df[[target_col]])

    def create_sequences(data, target, window=4, horizon=15):
        X, y = [], []
        for i in range(len(data) - window - horizon):
            X.append(data[i:i+window])
            y.append(target[i+window:i+window+horizon])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(
        train_df[feature_cols].values,
        train_df[target_col].values,
        window=4,
        horizon=15
    )
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    model = Sequential()

    model.add(Conv1D(filters=224, kernel_size=1, activation='relu',
                     input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv1D(filters=192, kernel_size=1, activation='relu'))
    model.add(Dropout(0.30))

    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))

    model.add(Dropout(0.10))
    model.add(Dense(15))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mae'
    )

    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("scalers", exist_ok=True)

    model.save(f"models/{safe_name}_model.keras")

    with open(f"scalers/{safe_name}_scalers.pkl", "wb") as f:
        pickle.dump({
            "scaler_X": scaler_X,
            "scaler_y": scaler_y
        }, f)


for district in districts:
    print(f"\nTraining model for {district.upper()}")
    train_for_district(district)
