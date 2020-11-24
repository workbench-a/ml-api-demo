from flask import Blueprint, request, jsonify
from simple_regression_demo.processing.local_data_management import load_dataset
from simple_regression_demo.predict import make_prediction
from simple_regression_demo import __version__ as _version
from api import __version__ as api_version

prediction_app = Blueprint('prediction_app', __name__)

@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        return 'ok'

# @prediction_app.route('/', methods=['GET'])
# def home():
#     if request.method == 'GET':
#         return 'Home Page'

@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                        'api_version': api_version})


@prediction_app.route('/v1/predict/regression', methods=['GET'])
def predict():
    if request.method == 'GET':
        # Step 1: Extract POST data from request body as JSON
        input_data = load_dataset(mode="production", file_name="test.csv")
        result = make_prediction(input_data=input_data)

        # Step 4: Convert numpy ndarray to list
        predictions = result.get('predictions').tolist()
        version = result.get('version')
        errors = ''

        # Step 5: Return the response as JSON
        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors})

# @prediction_app.route('/v1/predict/regression', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Step 1: Extract POST data from request body as JSON
#         input_data = load_dataset(mode="production", file_name="test.csv")
#         result = make_prediction(input_data=input_data)

#         # Step 4: Convert numpy ndarray to list
#         predictions = result.get('predictions').tolist()
#         version = result.get('version')
#         errors = ''

#         # Step 5: Return the response as JSON
#         return jsonify({'predictions': predictions,
#                         'version': version,
#                         'errors': errors})


