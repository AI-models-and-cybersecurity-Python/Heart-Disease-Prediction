import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import accuracy_score

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = pd.get_dummies(df, columns=['Sex', 'ST_Slope', 'ExerciseAngina', 'RestingECG', 'ChestPainType'])
    return df

def split_data(df):
    X = df.drop(["HeartDisease"], axis=1)
    Y = df['HeartDisease']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def standardize_data(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(6, activation='relu', input_dim=input_dim))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.33)
    return model

def make_predictions(model, x_test):
    preds = model.predict(x_test)
    preds = preds > 0.5
    return preds

def calculate_accuracy(y_test, preds):
    accuracy = accuracy_score(y_test, preds)
    return float(accuracy)

filepath = "heart.csv"
df = load_and_preprocess_data(filepath)
x_train, x_test, y_train, y_test = split_data(df)
x_train, x_test = standardize_data(x_train, x_test)
model = build_model(input_dim=x_train.shape[1])
model = train_model(model, x_train, y_train)
preds = make_predictions(model, x_test)
accuracy = calculate_accuracy(y_test, preds)

print(f"Model accuracy: {accuracy:.2f}")
