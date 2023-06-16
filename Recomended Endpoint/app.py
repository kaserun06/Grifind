import numpy as np
from flask import Flask , request, jsonify, render_template
import pickle
import tensorflow as tf
import joblib

app = Flask(__name__)

#load pickle model 
loaded_model = None

@app.route("/")
def Home():
    return 'hello world'

@app.route("/recommend", methods=['POST'])
def predict():
    # Get the request data
    data = request.json
    
    # Extract latitude and longitude from the request data
    user_id = data["user_id"]
    
    loaded = tf.saved_model.load(r'C:\Users\daffa\OneDrive\Desktop\model\griya_recommendation_model')

    scores, griya_names = loaded([str(user_id)])

    griya_names_array = griya_names.numpy()
    griya_names_list = griya_names_array.tolist()
    griya_names_string = str(griya_names_list[0][:3])
    
    response = {
        "user_id": user_id,
        "recommendations": griya_names_string
    }
    
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)

