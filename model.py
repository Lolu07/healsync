import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def train_model():
    np.random.seed(42)

    # Synthetic dataset (sleep hrs, stress 1-10, mood 1-5, screen time hrs)
    n = 500
    sleep = np.random.normal(6.5, 1.5, n)
    stress = np.random.randint(1, 11, n)
    mood = np.random.randint(1, 6, n)
    screentime = np.random.normal(4, 1.2, n)

    # Burnout risk label (simple rule-based pattern)
    risk = (sleep < 6).astype(int) + (stress > 6).astype(int) + (mood < 3).astype(int) + (screentime > 5).astype(int)
    label = (risk >= 2).astype(int)  # 1 = high burnout risk

    df = pd.DataFrame({
        "sleep": sleep,
        "stress": stress,
        "mood": mood,
        "screen": screentime,
        "label": label
    })

    X = df[["sleep", "stress", "mood", "screen"]]
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    with open("model.pkl", "wb") as f:
        pickle.dump((model, scaler), f)

    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()
