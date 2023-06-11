import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
from mlxtend.plotting import plot_decision_regions
import warnings
import os

app = Flask(__name__)
app.static_folder = 'static'

model = pickle.load(open("C:\\Users\\abdul\\OneDrive\\Desktop\\LAB\model\\mymodel_svm.pkl", "rb"))
scaler = pickle.load(open("C:\\Users\\abdul\\OneDrive\\Desktop\\LAB\\model\\scaler.pkl", "rb"))
data = pd.read_csv('C:\\Users\\abdul\\OneDrive\\Desktop\\LAB\model_code\\heart_disease_dataset.csv')

warnings.filterwarnings("ignore", category=UserWarning)


# defining the homepage
@app.route('/')
def home():
    default_data = [52, 1, 0, 125, 212, 0, 1, 168, 0, 1, 2, 2, 3]
    feature_names = data.drop('target', axis=1).columns
    feature_values = pd.Series(default_data, index=feature_names)
    return render_template("index.html", feature_values=feature_values)


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    print(features)  # Add this line to check the form data

    # Preprocess the input data
    input_data = np.array(features).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    if prediction == 1:
        result = 'Heart Disease Detected'
    else:
        result = 'No Heart Disease Detected'

    # Generate additional information
    feature_names = data.drop('target', axis=1).columns
    feature_values = pd.Series(features, index=feature_names)

    # Generate the box plot
    plt.figure(figsize=(12, 12))
    data.drop('target', axis=1).boxplot(grid=False)
    box_plots_path = os.path.join(app.static_folder, 'box_plots.png')
    plt.savefig(box_plots_path)
    plt.close()

    # Generate the histograms
    plt.figure(figsize=(12, 12))
    data.hist(figsize=(12, 12))
    histograms_path = os.path.join(app.static_folder, 'histograms.png')
    plt.savefig(histograms_path)
    plt.close()

    # Generate the correlation matrix
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
    correlation_matrix_path = os.path.join(app.static_folder, 'correlation_matrix.png')
    plt.savefig(correlation_matrix_path)
    plt.close()

    # Generate the scatter plot matrix
    scatter_plot_matrix = pd.plotting.scatter_matrix(data, figsize=(12, 12))
    scatter_plot_matrix_path = os.path.join(app.static_folder, 'scatter_plot_matrix.png')
    scatter_plot_matrix[0][0].figure.savefig(scatter_plot_matrix_path)
    plt.close()

    # Calculate the confusion matrix, accuracy, precision, recall, and F1 score
    X_test = data.drop('target', axis=1)
    y_test = data['target']
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    confusion_matrix_path = os.path.join(app.static_folder, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Generate the learning curve plot
    X_train = X_test  # Just for example, replace with your training data
    y_train = y_test  # Just for example, replace with your training data
    learning_curve_plot = learning_curve(model, X_train, y_train)
    plt.figure(figsize=(12, 10))
    plt.plot(learning_curve_plot[0], learning_curve_plot[1], label='Training score')
    plt.plot(learning_curve_plot[0], learning_curve_plot[2], label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend()
    learning_curve_path = os.path.join(app.static_folder, 'learning_curve.png')
    plt.savefig(learning_curve_path)
    plt.close()

    # Render the prediction result template and pass the additional information
    return render_template("prediction_result.html", result=result, feature_values=feature_values,
                           histograms='static/histograms.png', box_plots=box_plots_path,
                           correlation_matrix='static/correlation_matrix.png',
                           scatter_plot_matrix='static/scatter_plot_matrix.png',
                           confusion_matrix=confusion_matrix_path,
                           learning_curve=learning_curve_path,
                           accuracy=accuracy, precision=precision, recall=recall, f1=f1)


if __name__ == '__main__':
    app.run(debug=True)
