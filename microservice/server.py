from flask import Flask, jsonify, request
from joblib import load
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

# Cross Origin Resource Sharing (CORS) handling
CORS(app, resources={'/smart': {"origins": "http://localhost:8080"}})

model = load("fitted_random_forest.pkl")

@app.route('/smart', methods=['POST'])
def image_post_request():
    x = np.array(request.json['raw_smart'])
    y = model.predict_proba(x)[:, 1][0]
    failure = f"broke within 30d with prob: {y}" if y > 0.5 else f"good with prob: {1 - y}"

    return jsonify({'result': failure})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)