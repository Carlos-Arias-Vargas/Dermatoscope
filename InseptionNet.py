import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Definir las rutas de los datos
dirname = os.path.join(os.getcwd(), 'Train')
imgpath = dirname + os.sep

# Crear una lista vacía para almacenar las imágenes y los nombres de las clases
images = []
classes = []

print("leyendo imagenes de ",imgpath)

# Leer todas las imágenes y sus etiquetas de las carpetas
for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            images.append(image)
            classes.append(os.path.basename(root))

# Convertir las listas de Python a arrays de Numpy
images = np.array(images)
classes = np.array(classes)

# Obtener las etiquetas únicas
unique_classes = np.unique(classes)

# Convertir las etiquetas a números enteros
class_dict = {class_name: i for i, class_name in enumerate(unique_classes)}
labels = np.array([class_dict[class_name] for class_name in classes])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalizar los datos de imagen
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# Convertir las etiquetas a vectores one-hot
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Cargar la arquitectura InceptionNet pre-entrenada en ImageNet
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(100,100,3))

# Congelar todas las capas del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Añadir capas adicionales al modelo
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(len(unique_classes), activation='softmax')(x)

# Construir el modelo completo
model = Model(inputs=base_model.input, outputs=x)

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
epochs = 214
batch_size = 105
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))

# Evaluar el modelo con el conjunto de prueba
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', test_acc)

# Guardar el modelo en un archivo
model.save('InceptionNet.h5py')
