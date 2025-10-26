import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

#DATA COLLECTION AND ANALYSIS

diabetes_dataset = pd.read_csv('C:\\Users\\GOPISH\\Desktop\\ML_projects\\Diabetes_prediction\\diabetes.csv')
print(diabetes_dataset.head())

print(diabetes_dataset.shape)
print(diabetes_dataset.describe())
print(diabetes_dataset['Outcome'].value_counts())

print(diabetes_dataset.groupby('Outcome').mean())

X = diabetes_dataset.drop(columns='Outcome',axis=1)
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

#  SPLIT TRAINING AND TESTING DATA

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X_train)
print(Y_train)

# MODEL TRAINING 

classifier = svm.SVC(kernel='linear')
# training the Support Vector Machine Classifier

classifier.fit(X_train,Y_train)


# MODEL EVALUATION

# Accuracy score 
# Accuracy on training data

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print(f"Accuracy on training data : {training_data_accuracy}")

# Accuracy on test data

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print(f"Accuracy on test data : {test_data_accuracy}")

# MODEL PREDICTION 

input_data = (0,126,86,27,120,27.4,0.515,21)
columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

numpy_array = np.asarray(input_data)

reshaped_dataset = numpy_array.reshape(1,-1)

input_dataframe = pd.DataFrame(reshaped_dataset, columns=columns)  

prediction = classifier.predict(input_dataframe)

if (prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")

# PICKLING THE MODEL

pickle.dump(classifier,open('classifer_model.pkl','wb'))
loaded_model = pickle.load(open('classifer_model.pkl','rb'))

prediction= loaded_model.predict(input_dataframe)
if (prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")


