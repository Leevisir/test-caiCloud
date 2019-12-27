import os
import pandas as pd
from flask import Flask, request
from predict import predict_timeseries
from process_input import process_input
import json

app = Flask(__name__)

def _predict_message(historical, fitted, predictions):
    messagebody = {
        'historical': {
            'times': historical.x,
            'traffic': historical.y,
        },
        'fitted': {
            'times': fitted.x,
            'traffic': fitted.y,
        },
        'predicted': {
            'times': predictions.x,
            'traffic': predictions.y,
        },
    }
    
    return messagebody

@app.route('/', methods=['POST'])
def predict():
    try:
        file = request.files['file']
    except:
        response = app.response_class(
            response='Lacking data!',
            status=500
        )
        return response

    # load and preprocess input data
    df = pd.read_csv(file)
    data = process_input(df)

    # predict results
    historical, fitted, predictions = predict_timeseries(data)
    message_body = _predict_message(historical,fitted,predictions)

    # create response message
    response = app.response_class(
        response=json.dumps(message_body),
        mimetype='application/json'
    )
    return response



if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))