import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# Load the heart disease dataset
data = pd.read_csv('C:\\Users\\abdul\\OneDrive\\Desktop\\LAB\model_code\\heart_disease_dataset.csv')

# Preprocess the dataset
X = data.drop('target', axis=1)
y = data['target']
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the Support Vector Machine (SVM) model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)

# Save the model to a file
pickle.dump(svm_model, open("C:\\Users\\abdul\\OneDrive\\Desktop\\LAB\model\\mymodel_svm.pkl", "wb"))
# Save the scaler object to a file
pickle.dump(scaler, open("C:\\Users\\abdul\\OneDrive\\Desktop\\LAB\\model\\scaler.pkl", "wb"))

print("Success")
