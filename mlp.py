import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers.legacy import SGD as legacy_SGD

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255

# One-hot encoding of labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def build_model(optimizer, activation='relu', dropout_rate=0.0, batch_norm=False):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(784,)))
    if batch_norm:
        model.add(BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(512, activation=activation))
    if batch_norm:
        model.add(BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate(optimizer, activation, dropout_rate, batch_norm, epochs=20, batch_size=128):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model = build_model(optimizer, activation, dropout_rate, batch_norm)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_test, y_test), verbose=1,
                        callbacks=[early_stopping])

    return history


def plot_history(histories, titles):
    plt.figure(figsize=(12, 4 * len(histories)))
    for i, (history, title) in enumerate(zip(histories, titles)):
        plt.subplot(len(histories), 2, 2 * i + 1)
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title(f'{title} - Loss')
        plt.legend()

        plt.subplot(len(histories), 2, 2 * i + 2)
        plt.plot(history.history['accuracy'], label='train accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.title(f'{title} - Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.show()


# Train and evaluate with different optimizers and learning rates
histories = []
titles = []

# Example 1: Using Adam optimizer with ReLU activation
histories.append(train_and_evaluate(Adam(learning_rate=0.1), 'relu', dropout_rate=0.0, batch_norm=False))
titles.append('Adam - ReLU')

# Example 2: Using legacy SGD optimizer with learning rate decay and batch normalization
histories.append(train_and_evaluate(legacy_SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True), 'relu',
                                    dropout_rate=0.0, batch_norm=True))
titles.append('SGD (legacy) - ReLU with BatchNorm')

# Example 3: Using RMSprop optimizer with sigmoid activation and dropout
histories.append(train_and_evaluate(RMSprop(learning_rate=0.001), 'sigmoid', dropout_rate=0.5, batch_norm=False))
titles.append('RMSprop - Sigmoid with Dropout')

# Plot all results
plot_history(histories, titles)+