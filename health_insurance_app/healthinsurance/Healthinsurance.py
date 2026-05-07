import pickle
from pathlib import Path

import pandas as pd
import numpy as np


class HealthInsurance:
    def __init__(self):
        self.home_path = Path(__file__).resolve().parents[1]

        with open(self.home_path / "features" / "annual_premium_scaler.pkl", "rb") as file:
            self.annual_premium_scaler = pickle.load(file)

        with open(self.home_path / "features" / "age_scaler.pkl", "rb") as file:
            self.age_scaler = pickle.load(file)

        with open(self.home_path / "features" / "vintage_scaler.pkl", "rb") as file:
            self.vintage_scaler = pickle.load(file)

        with open(self.home_path / "features" / "target_encode_gender_scaler.pkl", "rb") as file:
            self.target_encode_gender_scaler = pickle.load(file)

        with open(self.home_path / "features" / "target_encode_region_code_scaler.pkl", "rb") as file:
            self.target_encode_region_code_scaler = pickle.load(file)

        with open(self.home_path / "features" / "fe_policy_sales_channel_scaler.pkl", "rb") as file:
            self.fe_policy_sales_channel_scaler = pickle.load(file)

    def data_cleaning(self, df1):
        df1 = df1.copy()
        df1.columns = df1.columns.str.lower()
        return df1

    def feature_engineering(self, df2):
        df2 = df2.copy()

        df2["vehicle_age"] = df2["vehicle_age"].apply(
            lambda x: "over_2_years"
            if x == "> 2 Years"
            else "between_1_2_years"
            if x == "1-2 Year"
            else "below_1_year"
        )

        df2["vehicle_damage"] = df2["vehicle_damage"].apply(
            lambda x: 1 if x == "Yes" else 0
        )

        return df2

    def data_preparation(self, df5):
        df5 = df5.copy()
    
        # garante que colunas que receberão valores decimais aceitem float
        df5["region_code"] = df5["region_code"].astype(float)
        df5["policy_sales_channel"] = df5["policy_sales_channel"].astype(float)
    
        # scalers
        df5["annual_premium"] = self.annual_premium_scaler.transform(df5[["annual_premium"]])
        df5["age"] = self.age_scaler.transform(df5[["age"]])
        df5["vintage"] = self.vintage_scaler.transform(df5[["vintage"]])
    
        # encoders
        df5["gender"] = df5["gender"].map(self.target_encode_gender_scaler).astype(float)
    
        df5["region_code"] = (
            df5["region_code"]
            .map(self.target_encode_region_code_scaler)
            .astype(float)
        )
    
        df5 = pd.get_dummies(df5, prefix="vehicle_age", columns=["vehicle_age"])
    
        df5["policy_sales_channel"] = (
            df5["policy_sales_channel"]
            .map(self.fe_policy_sales_channel_scaler)
            .astype(float)
        )
    
        col_selected = [
            "vintage",
            "annual_premium",
            "age",
            "region_code",
            "vehicle_damage",
            "policy_sales_channel",
            "previously_insured",
        ]
    
        return df5[col_selected]
    
    def get_prediction(self, model, original_data, test_data):
        pred = model.predict_proba(test_data)

        original_data = original_data.copy()
        original_data["score"] = pred[:, 1]

        return original_data.to_json(orient="records", date_format="iso")
