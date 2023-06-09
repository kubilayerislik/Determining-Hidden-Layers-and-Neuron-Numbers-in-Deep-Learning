import csv
import logging
from time import time

import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data(dataset_path):
    data = pd.read_excel(dataset_path)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    x = scaled_data[:, :14]
    y = scaled_data[:, 14:15]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test


def compile_model(input_shape):
    model = Sequential()
    model.add(Dense(775, activation="sigmoid", input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer="normal"))
    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=["mean_squared_error"]
    )
    return model


def train_and_score(dataset_path):
    x_train, x_test, y_train, y_test = get_data(dataset_path)
    input_shape = (x_train.shape[1],)
    model = compile_model(input_shape)
    early_stopper = EarlyStopping(patience=5)
    model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=1000,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[early_stopper],
    )
    score = model.evaluate(x_test, y_test, verbose=0)
    return score[1]


def train(dataset_path):
    logging.info("***Training ANN***")
    accuracy = train_and_score(dataset_path)
    logging.info("Average MSE: %.4f" % accuracy)
    return accuracy


def main():
    accuracy = train("data.xlsx")
    return accuracy


if __name__ == "__main__":
    accuracies = []
    times = []
    for _ in range(30):
        start_time = time()
        accuracy = main()
        time_cost = time() - start_time
        accuracies.append(accuracy)
        times.append(time_cost)

    with open("ann_result.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(zip(accuracies, times))
