import sys
from keras import models
import numpy as np
import matplotlib. pyplot as plt
model = models.load_model('InceptionNet.h5py')

x=sys.argv[1]
y=sys.argv[2]




r= "C:/Users/Carlos Arias/Documents/Carlos/IPN/9.Semestre IX/TT/Software/Pacientes/"

nueva_cadena = x.replace("_"," ")

images=[]
filenames = r+nueva_cadena+'/'+y+'/Blanco.jpg'


image = plt.imread(filenames)
images.append(image)

X = np.array(images, dtype=np.uint8) #convierto de lista a numpy
test_X = X.astype('float32')
test_X = test_X / 255.



predicted_classes = model.predict(test_X)
prediccion = predicted_classes # para ver la predicción de la primera foto
cadena = "".join(str(elemento) for elemento in prediccion)
cadena = cadena.replace("[","")
cadena = cadena.replace("]","")

arreglo_str = cadena.split()
arreglo_num = [float(elemento) for elemento in arreglo_str]
maximo = max(arreglo_num)
posicion = arreglo_num.index(maximo)

print(posicion)