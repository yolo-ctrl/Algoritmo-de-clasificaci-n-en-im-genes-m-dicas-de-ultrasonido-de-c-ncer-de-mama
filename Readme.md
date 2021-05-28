![N|Solid](https://www.ecured.cu/images/6/6b/Univ_ibague.jpg)

---
# Classification algorithm in breast cancer ultrasound medical imaging. 
---
This project was born with the objective of developing an algorithm capable of detecting the presence of breast cancer in medical images through the application of classification techniques based on machine learning, it should be noted that it is more focused for its execution from Google Collaboratory, in case of use a different software or directly from the Ubuntu terminal there are changes that must be made.
#
-----
## HARDWARE REQUIREMENTS:
---
- Procesador Intel Inside 
- 4Gb Ram
- Disco Duro Min 120Gb
- 
---
## SOFTWARE REQUIREMENTS:
---
- Python 3
- Kaggle.json
- [Breast Ultrasound Images Dataset](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset)
---
## OPERATING SYSTEM INSTALLATION & LIBRARIES
---
### OS installation:
---
-For the operating system it is enough to have windows 64 and access the online application Colaboratoy to run the code. otherwise you can use Ubuntu although, it is not mandatory.
if you want to have Ubuntu operating system, in the following [Website](https://ubuntu.com/download/desktop), you will find the step by step and the requirements for its installation.
---
### If you are going to use Ubuntun we recommend installing and updating libraries for the operating system.
---
Once we have the Ubuntu OS on the PC and we have an internet connection, we enter the console to update the libraries already installed at the factory, using the following instructions:

```sh
$ Ctrl + t
$ sudo apt-get uptgrade
```
Later we install the Python 3 library and its PIP add-on, which allows us to download the Github libraries, as follows:
```sh
$ sudo apt-get install python3
$ sudo apt-get install python3-pip
```
We proceed to install the [Sklearn library](https://scikit-learn.org/0.15/inshtmltall.), as follows:
```sh
$ pip3 install -U scikit-learn.
```
we install the [numpy library](https://pypi.org/project/numpy/) with the following instructions:
```sh
$ pip3 install numpy.
```
Next we install the [opencv library](https://pypi.org/project/opencv-python/) with the following instructions:
```sh
$ pip3 install opencv-python.
```

##  INSTALL KAGGLE AND IMPORT DATASET 
---
For goggle colaboratoriy with the following instructions: 
```sh
!pip install kaggle
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
!ls
import zipfile
zip_ref = zipfile.ZipFile("breast-ultrasound-images-dataset.zip", "r")
zip_ref.extractall("files")
print(zip_ref.namelist)
zip_ref.close()
```
For Ubuntu terminal or other software
Proceed to install [kaggle](https://www.kaggle.com/) with the command:
```sh
$ pip3 install kaggle.
```
Import dataset with the following instructions:
```sh
!chmod 600 ~/[file path]/kaggle.json
!kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
!ls
import zipfile
zip_ref = zipfile.ZipFile("breast-ultrasound-images-dataset.zip", "r")
zip_ref.extractall("files")
print(zip_ref.namelist)
zip_ref.close()
```
---
## DATA LABELING AND DATA CLASSIFICATION
---
For data labeling we generate two list , as follows:
```sh
direccion = r'/[file path]/Dataset_BUSI_with_GT/'
# crear listas vacías para contener las imágenes que se están leyendo

x_imagen = []
y_target = []

target = os.listdir(direccion)

for dirnombre in target:
  dir_archivo = os.path.join(direccion, dirnombre)
  for file in os.listdir(dir_archivo):
    nombre_arch = os.path.join(dir_archivo, file)
    #	print(nombre_arch)
    imagen = cv2.imread(nombre_arch,0)
    imagen = cv2.resize(imagen, (300,300))
    x_imagen.append(imagen.flatten())
    if (dirnombre == 'bening'):
      y_target.append(0)
    elif (dirnombre == 'malignant'):
      y_target.append(1)
    else:
      y_target.append(2)

x_imagen_n = np.array(x_imagen)
print(x_imagen_n.shape)

y_target_n = np.array(y_target)
print(y_target_n)
```
For the division of the data in training and test, the train_test_split component is used that facilitates the division of data by asking only the name of the containers, the percentage of data for the tests and a state for the division, in this case a random state, which generates a different order in each cycle, as shown in the following code:

```sh
# dividir datos de entrenamiento y datos de prueba
x_train, x_test, y_train, y_test = train_test_split(x_imagen_n, y_target_n, test_size=0.3, random_state=0)
```
For the training and testing of the classifier the following codes are used which show different types of classifiers that can be used.

```sh
# entrenamiento del modelo de clasificacion randomforest
clf_rfc = RandomForestClassifier().fit(x_train, y_train)

# evaluar el modelo con datos de prueba
y_pred_rfc = clf_rfc.predict(x_test)

# entrenamiento del modelo de clasificacion arbol de deciciones 
algoritmo = DecisionTreeClassifier()
algoritmo.fit(x_train, y_train)

# evaluar el modelo con datos de prueba
Y_pred = algoritmo.predict(x_test)

# entrenamiento del modelo de clasificacion SVC
algoritmo = SVC()
algoritmo.fit(x_train, y_train)

# evaluar el modelo con datos de prueba
Y_pred2 = algoritmo.predict(x_test)
print('Precisión SVC: {}'.format(algoritmo.score(x_train, y_train)))
```
After training and testing, a classification accuracy check is performed using the F1 score, using the following lines of code:

```sh
F11 = f1_score(y_test, Y_pred.T, average='micro')

print("F11:",F11*100 ,"%")
PR1=precision_recall_fscore_support(y_test,Y_pred.T)
print(PR1)
```
---
# CODE
> python3 algoritmo_deteccion_cancer_mama.py
---
#### Authors:
----
- Manuel Felipe Sarmiento Gonzalez. 2420142004
- Leisson Duwer Polo Villalobos 2420171094

## Tutor:
- [Harold F. Murcia Moreno](http://haroldmurcia.com/) 


---