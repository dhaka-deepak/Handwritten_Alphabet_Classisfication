import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt


class HandwrittenCharacterRecognition:
    def __init__(self):
        self.model = None
        self.word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                          12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                          23: 'X', 24: 'Y', 25: 'Z'}

    def load_data(self, csv_path):
        """
        Load the data from the specified CSV file and preprocess it.
        """
        data = pd.read_csv(csv_path).astype('float32')
        X = data.drop('0', axis=1)
        y = data['0']
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
        train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28))
        test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28))
        self.train_X = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
        self.test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
        self.train_y = np_utils.to_categorical(train_y, num_classes=26, dtype='int')
        self.test_y = np_utils.to_categorical(test_y, num_classes=26, dtype='int')

    def create_model(self):
        """
        Create the Convolutional Neural Network (CNN) model architecture.
        """
        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(26, activation="softmax"))
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        """
        Train the model using the training data and evaluate its performance on the validation data.
        """
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
        history = self.model.fit(self.train_X, self.train_y, epochs=1, callbacks=[reduce_lr, early_stop],
                                 validation_data=(self.test_X, self.test_y))
        return history

    def save_model(self, model_path):
        """
        Save the trained model to the specified file path.
        """
        self.model.save(model_path)

    def load_model(self, model_path):
        """
        Load a pre-trained model from the specified file path.
        """
        self.model = load_model(model_path)

    def predict(self, test_samples):
        """
        Make predictions on the given test samples using the trained model.
        """
        predictions = self.model.predict(test_samples)
        return predictions

    def display_predictions(self, test_samples, predictions):
        """
        Display the predicted characters and their corresponding images.
        """
        fig, axes = plt.subplots(3, 3, figsize=(8, 9))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            img = np.reshape(test_samples[i], (28, 28))
            ax.imshow(img, cmap="Greys")
            pred = self.word_dict[np.argmax(predictions[i])]
            ax.set_title("Prediction: " + pred)
            ax.grid()
        plt.show()


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Character Recognition")
        self.geometry("400x200")

        self.hcr = HandwrittenCharacterRecognition()
        self.train_history = None

        self.style = ttk.Style(self)
        self.style.configure('TButton', font=('Arial', 12), padding=10)

        self.train_button = ttk.Button(self, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        self.predict_button = ttk.Button(self, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)

    def train_model(self):
        """
        Train the model by loading the CSV file, creating the model, and saving the trained model.
        """
        csv_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if csv_path:
            self.hcr.load_data(csv_path)
            self.hcr.create_model()
            self.train_history = self.hcr.train_model()
            self.hcr.save_model("model_hand.h5")
            messagebox.showinfo("Training Complete", "Model trained successfully!")

    def predict(self):
        """
        Predict the characters by loading a pre-trained model and displaying the predictions.
        """
        model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.h5")])
        if model_path:
            self.hcr.load_model(model_path)
            test_samples = self.hcr.test_X[:9]
            predictions = self.hcr.predict(test_samples)
            self.hcr.display_predictions(test_samples, predictions)


if __name__ == "__main__":
    app = Application()
    app.mainloop()
