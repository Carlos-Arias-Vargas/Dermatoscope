# importing the libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = keras.datasets.fashion_mnist.load_data()

print("X_train: ", X_train_raw.shape)
print("y_train: ", y_train_raw.shape)
print("X_test:  ", X_test_raw.shape)
print("y_test:  ", y_test_raw.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

X_train_raw[0][:5]

plt.imshow(X_train_raw[0], cmap=plt.cm.binary)
plt.colorbar()
plt.show()

print("Class label: ", y_train_raw[0])
print("Class name:  ", class_names[y_train_raw[0]])

plt.figure(figsize=(16,6))
for i in range(24):
    plt.subplot(3,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train_raw[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train_raw[i]])
plt.show()

def normalize(X):
  X = (X / 255.0).astype('float32')
  return X
     
X_train = normalize(X_train_raw)
X_test = normalize(X_test_raw)

# redimensionar as imagens
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

print("X_train: ", X_train.shape)
print("X_test:  ", X_test.shape)
     
X_train:  (60000, 28, 28, 1)
X_test:   (10000, 28, 28, 1)

np.unique(y_train_raw)

y_train = keras.utils.to_categorical(y_train_raw,10)
y_test = keras.utils.to_categorical(y_test_raw,10)

print("First Label Before One-Hot Encoding: ", y_train_raw[0])
print("First Label After One-Hot Encoding:  ", y_train[0])

model = keras.models.Sequential([
  keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', 
                      padding='same', input_shape=[28, 28, 1]),
  keras.layers.MaxPool2D(pool_size=2),
  keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', 
                      padding='same'),
  keras.layers.MaxPool2D(pool_size=2),
  keras.layers.Flatten(),
  keras.layers.Dense(units=128, activation='relu'),
  keras.layers.Dropout(0.25),
  keras.layers.Dense(units=64, activation='relu'),
  keras.layers.Dropout(0.25),
  keras.layers.Dense(units=10, activation='softmax'),
])
   
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_history = model.fit(X_train, y_train, batch_size=50, epochs=10, validation_split=0.3)

pd.DataFrame(model_history.history).plot()
plt.show()

model.evaluate(X_test, y_test);

predictions = model.predict(X_test)

def plot_img_label(img, pred_class, pred_percentage, true_class):

  plt.imshow(img,cmap=plt.cm.binary)

  if pred_class == true_class:
    color = 'blue'
  else:
    color = 'red'

  plt.title(label= f"Predicted: {pred_class} - {pred_percentage:2.1f}%\nActual: {true_class}", 
            fontdict={'color': color})
  
  plt.figure(figsize=(14,10))
for i in range(20):
    
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    i = i * 5

    img = X_test[i].reshape(28,28)
    pred_class = class_names[np.argmax(predictions[i])]
    pred_percentage = np.max(predictions[i])*100
    true_class = class_names[np.argmax(y_test[i])]
    
    plot_img_label(img, pred_class, pred_percentage, true_class)

plt.tight_layout()
plt.show()

predicted_label = np.argmax(predictions,axis = 1)
true_label = np.argmax(y_test, axis = 1)

crosstab = pd.crosstab(true_label, predicted_label, rownames=["True"], colnames=["Predicted"], margins=True)

classes = {}
for item in zip(range(10), class_names):
  classes[item[0]] = item[1]
  
crosstab.rename(columns=classes, index=classes, inplace=True)
crosstab