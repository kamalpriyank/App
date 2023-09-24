import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify

#Creating Toy Data for model 
X = np.array([9,8,7,1,2]).reshape(-1,1)
y = np.array([18,16,14,2,4])

#Create & Train Model

model = LinearRegression()
model.fit(X,y)

model_filename = 'linear_regression_model.joblib'
joblib.dump(model,model_filename)

from flask import Flask, request, jsonify
import joblib

#Create App

app = Flask(__name__)

#Load Model

model_filename = 'linear_regression_model.joblib'
model = joblib.load(model_filename)


@app.route('/')
def index():
    return "Simple Linear Regression Model API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        prediction = model.predict(np.array(data).reshape(-1, 1))
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
