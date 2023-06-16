import numpy as np
from flask import Flask , request, jsonify, render_template
import pickle
import joblib

app = Flask(__name__)

#load pickle model 
loaded_model = None

@app.route("/")
def Home():
    return 'hello world'

@app.route("/predict", methods=['POST'])
def predict():
    # Get the request data
    data = request.json
    
    # Extract latitude and longitude from the request data
    user_latitude = data["latitude"]
    user_longitude = data["longitude"]
    
    # Convert latitude and longitude to numpy array
    user_location = np.array([[user_latitude, user_longitude]])
    loaded_model = joblib.load(r'C:\Users\daffa\OneDrive\Desktop\model\knn_model.pkl') #<<<ini diganti pathnya...

    # loaded_model = load_model()
    nearest_griya = loaded_model.predict(user_location)
    
   
    # Return the nearest griya as a response
    response = {
        "nearest_griya": nearest_griya.tolist()
    }
    
    return jsonify(response)

# @app.route("/predict", methods=['POST'])
# def predict_nearest_griya():
#     # Get the request data
#     data = request.json
    
#     # Extract latitude and longitude from the request data
#     user_id = data[user_id]
    
#     # Convert user_id to numpy array
#     query_point = np.array([user_id])
    
#     # Use the model to find the nearest griya
#     recommended_griya = model_griya_reco.predict(query_point)
    
#     # Return the recommended griya as a response
#     response = {
#         "nearest_hotel": recommended_griya[0]
#     }
    
#     return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)

