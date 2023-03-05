"""
testing the model output quality by flask
"""
from flask import Flask, jsonify, request
from model_inference import TestModel
__author__ = "Maryam Najafi"
__organization__ = "AuthorShip Attribution"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "11/14/2021"
APP = Flask(__name__)


# routes
@APP.route('/', methods=['POST', 'GET'])
def predict():
    """
    to test the quality of target prediction
    """
    data = request.get_json(force=True)
    ouput_dict = INS.run_flask(data)

    return jsonify(ouput_dict)


if __name__ == "__main__":
    INS = TestModel()
    APP.run(debug=True, host="0.0.0.0", port=8027, threaded=True)
