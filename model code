# -*- coding: utf-8 -*-
"""
Created on Sat May 17 15:10:19 2025

@author: fawad
"""

#Open Dataset File

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
#from google.colab import files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from urllib.parse import unquote
import seaborn as sns
import numpy as np

def normalize_payload(payload):
    payload = unquote(payload)  # decode URL encoding
    payload = payload.lower()
    payload = payload.replace('//', '/')
    payload = payload.replace('%00', '')
    return payload

#uploaded = files.upload()

#file_path = 'C:/Users/fawad/OneDrive/Documents/File-Inclusion-Attacks-Dataset/file-inclusion_combined.csv'
df = pd.read_csv('file_inclusion_with_benign.csv')
print(df)

#Preprocess Dataset

#Remove Empty Values
df = df.dropna()
print(df)

#Remove Duplicates
df = df.drop_duplicates()
print(df)

#Vectorize the entire dataset before splitting
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 7), max_features=2000)
# Fit the vectorizer on the training data and transform both training and testing data
df['Sentence'] = df['Sentence'].apply(normalize_payload)
X_vec= vectorizer.fit_transform(df['Sentence'])
print("This is vectorized X", X_vec)

# Convert to dense array to combine with labels
X_dense = X_vec.toarray()

#Get names of features
feature_names = vectorizer.get_feature_names_out()
print("Feature Names:", feature_names)

# Combine vectorized features with labels into a DataFrame
X = pd.DataFrame(X_dense, columns=feature_names)
X['Label'] = df['Label'].values

# Make Label column the target and load it into y
y = X['Label']

# Now drop the Label column because it's the target
X = X.drop('Label', axis=1)
# Now split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Print Shapes of New X Objects
print("X_train Shape:",  X_train.shape)
print("X_test Shape:", X_test.shape)

#Printing the shapes of the new y objects
print("Y_train Shape:", y_train.shape)
print("Y_test Shape: ",y_test.shape)

#Logistic Regression

# Initialize the logistic regression model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, accuracy_score

# Initialize a TfidfVectorizer to convert text to numerical features
#vectorizer = TfidfVectorizer()

#X_train_vec = vectorizer.fit_transform(X_train)
#X_test_vec = vectorizer.transform(X_test)
logistic_model = LogisticRegression(class_weight='balanced')
logistic_model.fit(X_train, y_train)

logistic_train = logistic_model.predict(X_train)
logistic_test = logistic_model.predict(X_test)

logistic_train_mse = mean_squared_error(y_train, logistic_train)
logistic_train_score = accuracy_score(y_train, logistic_train)

logistic_test_mse = mean_squared_error(y_test, logistic_test)
logistic_test_score = accuracy_score(y_test, logistic_test)

print("Logistic Regression model accuracy (in %):", logistic_test_score*100)

print('Logistic MSE (Train):', logistic_train_mse)
print('Logistic R2 (Train):', logistic_train_score)
print('Logistic MSE (Test):', logistic_test_mse)
print('Logistic R2 (Test):', logistic_test_score)

# Find this model's F-1 score
logistic_f1_score = metrics.f1_score(y_test, logistic_test)
print('Logistic F1 Score:', logistic_f1_score)

# Find this model's AUC
logistic_auc_score = metrics.roc_auc_score(y_test, logistic_test)
print('Logistic AUC:', logistic_auc_score)

# Find this model's precision
logistic_precision = metrics.precision_score(y_test, logistic_test)
print('Logistic Precision:', logistic_precision)

# Find Recall
logistic_recall = metrics.recall_score(y_test, logistic_test)
print('Logistic Recall:', logistic_recall)

logistic_results = pd.DataFrame(['Logistic Regression', logistic_train_mse, logistic_train_score, logistic_test_mse, logistic_test_score]).transpose()
logistic_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
logistic_results

#y_pred = clf.predict(X_test_vec)

#Evaluate Performance of Logistic Regression
#acc = accuracy_score(y_test, y_pred)
#print("Logistic Regression model accuracy (in %):", acc*100)

#Random Forest Classifier


#Use Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

rand_forest_model = RandomForestClassifier(n_estimators=40, class_weight='balanced', random_state=42)

# Use dataset for training
rand_forest_model.fit(X_train, y_train)

# Make a prediction
model_prediction1 = rand_forest_model.predict(X_train)
model_prediction2 = rand_forest_model.predict(X_test)

train_accuracy = accuracy_score(y_train, model_prediction1)
print("Training", f"Accuracy: {train_accuracy}")
test_accuracy = accuracy_score(y_test, model_prediction2)
print("Testing", f"Accuracy: {test_accuracy}")

#Implement an F-1 score
rand_forest_f1_score = metrics.f1_score(y_test, model_prediction2)
print('Random Forest F1 Score:', rand_forest_f1_score)

#Find this model's AUC
rand_forest_auc_score = metrics.roc_auc_score(y_test, model_prediction2)
print('Random Forest AUC:', rand_forest_auc_score)

#Find this model's precision
rand_forest_precision = metrics.precision_score(y_test, model_prediction2)
print("Random forest precision is", rand_forest_precision)

# Find Recall
rand_forest_recall = metrics.recall_score(y_test, model_prediction2)
print('Random Forest Recall:', rand_forest_recall)



#Decision Tree Classifier

from sklearn import tree
decision_tree_model = tree.DecisionTreeClassifier(class_weight='balanced')
decision_tree_model = decision_tree_model.fit(X_train, y_train)
decision_tree_prediction1 = decision_tree_model.predict(X_train)
decision_tree_prediction2 = decision_tree_model.predict(X_test)

decision_tree_train_accuracy = accuracy_score(y_train, decision_tree_prediction1)
print("Training", f"Accuracy: {decision_tree_train_accuracy}")
decision_tree_test_accuracy = accuracy_score(y_test, decision_tree_prediction2)
print("Testing", f"Accuracy: {decision_tree_test_accuracy}")

#Implement an F-1 score
decision_tree_f1_score = metrics.f1_score(y_test, decision_tree_prediction2)
print('Decision Tree F1 Score:', decision_tree_f1_score)

# Find this model's AUC and precision
decision_tree_auc_score = metrics.roc_auc_score(y_test, decision_tree_prediction2)
print('Decision Tree AUC:', decision_tree_auc_score)
decision_tree_precision = metrics.precision_score(y_test, decision_tree_prediction2)
print('Decision Tree Precision:', decision_tree_precision)

# Find Recall
decision_tree_recall = metrics.recall_score(y_test, decision_tree_prediction2)
print('Decision Tree Recall:', decision_tree_recall)


#K-Nearest Neighbor

K = []
training = []
test = []
scores = {}

for k in range(2, 21):
    KNN_model = KNeighborsClassifier(n_neighbors = k)
    KNN_model.fit(X_train, y_train)

    training_score = KNN_model.score(X_train, y_train)
    test_score = KNN_model.score(X_test, y_test)
    K.append(k)

    training.append(training_score)
    test.append(test_score)
    scores[k] = [training_score, test_score]

#Evaluate KNN Model
for keys, values in scores.items():
    print(keys, ':', values)

#Plot the Scores
ax = sns.stripplot(x=K, y=training);
ax.set(xlabel ='values of k', ylabel ='Training Score')
plt.show()
#function for showing the plot
ax = sns.stripplot(x=K, y=test);
ax.set(xlabel ='values of k', ylabel ='Test Score')
plt.show()

# Overlapping plots
plt.scatter(K, training, color ='k')
plt.scatter(K, test, color ='g')
plt.show()

#Implement an F-1 score
KNN_f1_score = metrics.f1_score(y_test, KNN_model.predict(X_test))
print('KNN F1 Score:', KNN_f1_score)

# Find this model's AUC
KNN_auc_score = metrics.roc_auc_score(y_test, KNN_model.predict(X_test))
print('KNN AUC:', KNN_auc_score)
# Find this model's precision
KNN_precision = metrics.precision_score(y_test, KNN_model.predict(X_test))
print('KNN Precision:', KNN_precision)

# Find Recall
KNN_recall = metrics.recall_score(y_test, KNN_model.predict(X_test))
print('KNN Recall:', KNN_recall)



#Support Vector Machine


# Implement a linear SVM
from sklearn.svm import LinearSVC

svm_model = LinearSVC(random_state=0, class_weight='balanced')

# Train the SVM
svm_model.fit(X_train, y_train)

# Make Prediction
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy}")

# Find this model's F1 score
svm_f1_score = metrics.f1_score(y_test, y_pred)
print('SVM F1 Score:', svm_f1_score)

# Find this model's precision
svm_precision = metrics.precision_score(y_test, y_pred)
print('SVM Precision:', svm_precision)

# Find this model's AUC
svm_auc_score = metrics.roc_auc_score(y_test, y_pred)
print('SVM AUC:', svm_auc_score)

# Find Recall
svm_recall = metrics.recall_score(y_test, y_pred)
print('SVM Recall:', svm_recall)

# Make Decision Boundary
def plot_decision_boundary(model, X, y, ax):
    # Reduce X to 2 dimensions using PCA for visualization purposes
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_new = pca.fit_transform(X.values)  # Convert X to dense array

    x_min, x_max = X_new[:, 0].min() - 1, X_new[:, 0].max() + 1
    y_min, y_max = X_new[:, 1].min() - 1, X_new[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Use PCA to transform the grid data
    grid_data = np.c_[xx.ravel(), yy.ravel()]
    grid_data_original = pca.inverse_transform(grid_data)

    Z = model.predict(grid_data_original)

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X_new[:, 0], X_new[:, 1], c=y, s=20, cmap='RdBu')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plot_decision_boundary(svm_model, X_train, y_train, ax)
plt.title("Linear SVM Decision Boundary")
plt.show()




#Neural Network (RNN)

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

#Neural Network Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
neural_model = MLPClassifier(activation='relu', solver='adam', alpha=1e-4, max_iter=900,
                             hidden_layer_sizes=(64, 32), random_state=1)
MLPClassifier(alpha=1e-04, hidden_layer_sizes=(64, 32), random_state=1,
              solver='adam')
neural_model.fit(X_train_scaled, y_train)
neural_prediction1 = neural_model.predict(X_train_scaled)
neural_prediction2 = neural_model.predict(X_test_scaled)

neural_train_accuracy = accuracy_score(y_train, neural_prediction1)
print("Training", f"Accuracy: {neural_train_accuracy}")
neural_test_accuracy = accuracy_score(y_test, neural_prediction2)
print("Testing", f"Accuracy: {neural_test_accuracy}")

#Implement an F-1 score
neural_f1_score = metrics.f1_score(y_test, neural_prediction2)
print('Neural Network F1 Score:', neural_f1_score)

# Find this model's AUC
neural_auc_score = metrics.roc_auc_score(y_test, neural_prediction2)
print('Neural Network AUC:', neural_auc_score)

# Find this model's precision
neural_precision = metrics.precision_score(y_test, neural_prediction2)
print('Neural Network Precision:', neural_precision)

# Find Recall
neural_recall = metrics.recall_score(y_test, neural_prediction2)
print('Neural Network Recall:', neural_recall)


#Cross Validation on Models

from sklearn.metrics import roc_auc_score
from sklearn.base import clone

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": rand_forest_model,
    "Decision Tree": decision_tree_model,
    "KNN": KNN_model,
    "SVM": svm_model,
    "Neural Network": neural_model
}

k = 10  # Number of folds
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Save the vectorizer as a joblib file
print("Save the vectorizer as a joblib file first")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

for name, model in models.items():
    current_model = model
    scores = cross_val_score(model, X_test, y_test, cv=cv, scoring="roc_auc")  # Use different scoring metrics as needed
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std Dev = {scores.std():.4f}")

    # Save each model as a joblib file
    filename = f'model_{name}.joblib'
    joblib.dump(current_model, filename)
    print(f"Model {name} saved as {filename}")
