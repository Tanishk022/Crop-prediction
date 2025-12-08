# from flask import Flask, render_template, request
# import numpy as np
# import pickle

# app = Flask(__name__)

# model = pickle.load(open("crop_model_1.pkl", "rb"))

# @app.route('/')
# def home():
#     # return render_template("index.html")
#     return render_template("predict.html")


# @app.route('/predict', methods=['POST'])
# def predict():

#     N = float(request.form['N'])
#     P = float(request.form['P'])
#     K = float(request.form['K'])
#     ph = float(request.form['ph'])
#     rainfall = float(request.form['rainfall'])

#     features = np.array([[N, P, K, ph, rainfall]])

#     result = model.predict(features)[0]

#     return render_template("result.html", crop=result)

# if __name__ == "__main__":
#     app.run(debug=True)
 

from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("crop_model_1.pkl", "rb"))
# model = pickle.load(open("crop_model_1.pkl", "rb"))


# Home Page
@app.route('/')
def home():
    return render_template("index.html")

# Predict Input Page
@app.route('/predict')
def predict_page():
    return render_template("predict.html")

# Contact Page
@app.route('/contact')
def contact_page():
    return render_template("contact.html")

# Prediction Handler
@app.route('/get_prediction', methods=['POST'])
def get_prediction():

    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    result = model.predict(features)[0]

    return render_template("result.html", crop=result)

# @app.route('/get_prediction', methods=['POST'])
# def get_prediction():
#     N = float(request.form['N'])
#     P = float(request.form['P'])
#     K = float(request.form['K'])
#     ph = float(request.form['ph'])
#     rainfall = float(request.form['rainfall'])

#     features = np.array([[N, P, K, ph, rainfall]])
#     result = model.predict(features)[0]

#     return render_template("result.html", crop=result)

if __name__ == "__main__":
    app.run(debug=True)

# print(model)
