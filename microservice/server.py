from flask import Flask, jsonify, request
from joblib import load
import numpy as np
from flask_cors import CORS
from xbg import DMatrix

app = Flask(__name__)

# Cross Origin Resource Sharing (CORS) handling
CORS(app, resources={'/smart': {"origins": "http://localhost:8080"}})

model = load("fitted_xgb.pkl")

@app.route('/smart', methods=['POST'])
def image_post_request():
    x = np.array(request.json['raw_smart'])
    smart_192cumul = x[-7] / x[1]
    smart_240_div_9 = x[-3] / x[1]
    smart_197_div_241 = x[-6] / x[-2]
    smart_198_div_241 = x[-5] / x[-2]
    smart_242_div_241 = x[-1] / x[-2]
    data = np.concatenate([
        x, smart_192cumul, smart_240_div_9, smart_197_div_241,
        smart_198_div_241, smart_242_div_241
    ])
    y = model.predict(DMatrix(data))[0]
    failure = f"broke within 30d with prob: {y}" if y > 0.5 else f"good with prob: {round(1 - y, 5)}"

    return jsonify({'result': failure})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)