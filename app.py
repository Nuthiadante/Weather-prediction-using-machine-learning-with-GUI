from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv(r'C:\Users\DK\Documents\Weather\weather.csv')

# Prepare data for model
X = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the data from the request
        data = request.get_json()

        # Extract the MinTemp value
        min_temp = data['MinTemp']

        # Make prediction
        max_temp_pred = model.predict([[min_temp]])

        # Prepare response
        response = {
            'MinTemp': min_temp,
            'MaxTemp_Predicted': float(max_temp_pred[0])
        }

        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
