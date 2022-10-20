import pickle
import flask
from flask import Flask, request, app, jsonify, render_template

import numpy as np
import pandas as pd

app = Flask(__name__)

#Load Pickle file
model = pickle.load(open('decision_tree.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

#Postman
@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    new_data = [list(data.values())]
    predict = model.predict(new_data)
    if predict[0] == 0:
        return jsonify('No Fire')
    else:
        return jsonify('Fire')


#Webpage
@app.route('/predict', methods = ['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        new_data = np.array(data).reshape(1,-1)
        prediction = model.predict(new_data)
        if prediction[0] == 0:
            return render_template('home.html',prediction_text = 'No fire')
        else:
            return render_template('home.html', prediction_text = 'Fire')
    except:
       return render_template('home.html', prediction_text = 'Invalid Input') 

if __name__ == '__main__':
    app.run(debug = True)