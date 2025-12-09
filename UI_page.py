from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__, template_folder="templates", static_folder="static")


def load_or_train_model():
    """Load a trained model; train and save one if the pickle holds a string or is missing."""
    try:
        with open("crop_model_1.pkl", "rb") as f:
            mdl = pickle.load(f)
        # Validate the loaded object
        if hasattr(mdl, "predict"):
            return mdl
    except FileNotFoundError:
        mdl = None

    # Fallback: train a fresh RandomForest on the cleaned dataset
    df = pd.read_csv("clean_crop_data.csv")
    X = df.drop("label", axis=1)
    y = df["label"]

    trained_model = RandomForestClassifier(n_estimators=200, random_state=42)
    trained_model.fit(X, y)

    # Persist for future runs
    with open("crop_model_1.pkl", "wb") as f:
        pickle.dump(trained_model, f)

    return trained_model


model = load_or_train_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    N = float(request.form.get("N", 0))
    P = float(request.form.get("P", 0))
    K = float(request.form.get("K", 0))
    temperature = float(request.form.get("temperature", 0))
    humidity = float(request.form.get("humidity", 0))
    ph = float(request.form.get("ph", 0))
    rainfall = float(request.form.get("rainfall", 0))

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    result = model.predict(features)[0]

    return render_template("result.html", crop=result)


if __name__ == "__main__":
    app.run(debug=True)
 