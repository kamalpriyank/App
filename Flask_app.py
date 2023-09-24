from flask import Flask, request, jsonify
from flask_restful import Api
import joblib

app = Flask(__name__)

model_filename = 'linear_regression_model.joblib'
model = joblib.load(model_filename)

api = Api(app)

class PredictResource(Resource):

    def post(self):
        data = request.get_json()
        prediction = model.predict(np.array(data).reshape(-1, 1))
        return jsonify({'prediction': prediction.tolist()})
api.add_resource(PredictResource, '/predict')

if __name__ == '__main__':
    app.run(debug=True)