import itertools
import matplotlib
matplotlib.use("TKAgg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import *
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Building the model : neural network with L1 regularization
def MyModel(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(128, input_dim=n_inputs, kernel_regularizer=l1(0.005), activation='relu'))
    model.add(Dense(256, kernel_regularizer=l1(0.005), activation='relu'))
    model.add(Dense(256, kernel_regularizer=l1(0.005), activation='relu'))
    model.add(Dense(512, kernel_regularizer=l1(0.005), activation='relu'))
    model.add(Dense(n_outputs, kernel_regularizer=l1(0.005), activation='linear'))

    # Compiling the model
    model.compile(loss='mse', optimizer='adam', metrics=[mean_squared_error])

    # model summary
    model.summary()

    return model


def evaluation(model, X_test, y_test):
    # MSE and r squared values
    y_pred = model.predict(X_test)
    print("r2 score:", round(r2_score(y_test, y_pred), 4))
    # Scatter plot of predicted values
    plt.scatter(y_test, y_pred, s=2, alpha=0.7)
    plt.plot(list(range(2, 8)), list(range(2, 8)), color='black', linestyle='--')
    plt.title('Predicted vs. actual values of Test set')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.show()
