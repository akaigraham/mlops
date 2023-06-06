import pickle
from flask import Flask, request, jsonify

# load model
with open('lin_reg.bin', 'rb') as f_in:
    dv,model = pickle.load(f_in)
    
# preprocess features
def prepare_features(ride):
    features = {}
    features["PU_DO"] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features["trip_distance"] = ride['trip_distance'] # no feature engineering
    return features
    
# function to predict 
def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[0]

# create Flask app
app = Flask('duration-prediction')

@app.route('/predict', methods=['POST']) # turn flask app into endpoint    
def predict_endpoint():
    
    # get ride and pass it
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)
    
    result = {
        'duration': pred
    }
    
    # turn dictionary into json
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)