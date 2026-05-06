import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, request, Response

from healthinsurance.Healthinsurance import HealthInsurance


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model_health_insurance.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return Response('{"status": "ok"}', status=200, mimetype="application/json")


@app.route("/health", methods=["GET"])
def health():
    return Response('{"status": "healthy"}', status=200, mimetype="application/json")


@app.route("/predict", methods=["POST"])
def health_insurance_predict():
    test_json = request.get_json()

    if not test_json:
        return Response("{}", status=200, mimetype="application/json")

    if isinstance(test_json, dict):
        test_raw = pd.DataFrame(test_json, index=[0])
    else:
        test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

    pipeline = HealthInsurance()

    df1 = pipeline.data_cleaning(test_raw)
    df2 = pipeline.feature_engineering(df1)
    df3 = pipeline.data_preparation(df2)

    df_response = pipeline.get_prediction(model, test_raw, df3)

    return Response(df_response, status=200, mimetype="application/json")


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)