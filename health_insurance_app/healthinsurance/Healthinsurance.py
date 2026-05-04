import pickle
import pandas as pd
import numpy as np

class Healthinsurance:
    def __init__(self):
        self.home_path = '/home/ds-eduardo/propensao_classificacao/'
        self.annual_premium_scaler =                 pickle.load(open(self.home_path + 'src/features/annual_premium_scaler.pkl'))
        self.age_scaler =                            pickle.load(open(self.home_path + 'src/features/age_scaler.pkl'))
        self.vintage_scaler =                        pickle.load(open(self.home_path + 'src/features/vintage_scaler.pkl'))
        self.target_encode_gender_scaler =           pickle.load(open(self.home_path + 'src/features/target_encode_gender_scaler.pkl'))
        self.target_encode_region_code_scaler =      pickle.load(open(self.home_path + 'src/features/target_encode_region_code_scaler.pkl'))
        self.fe_policy_sales_channel_scaler =        pickle.load(open(self.home_path + 'src/features/fe_policy_sales_channel_scaler.pkl'))
    
    def data_cleaning(df1):

        #rename coloumns
        df_raw.columns = df_raw.columns.str.lower()

        #rename
        df1 = df_raw
        return df1

    def feature_engineering(df2):

        # vehicle age
        df2['vehicle_age'] = df2['vehicle_age'].apply(lambda x:'over_2_years' if x== '> 2 Years' else 'between_1_2_years' if x =='1-2 Year' else 'below_1_year')
        # vehicle damage
        df2['vehicle_damage'] = df2['vehicle_damage'].apply(lambda x:1 if x =='Yes' else 0)

        return df2


    def data_preparation(df5):

        #annual_premium
        df5["annual_premium"] = self.annual_premium_scaler.transform(df5[["annual_premium"]])

        #age
        df5["age"] = self.age_scaler.transform(df5[["age"]])

        #vintage
        df5["vintage"] = self.vintage_scaler.transform(df5[["vintage"]])

        ## 5.3 Encoder
        # gender - One Hot Encoding / Target Encoding
        df5['gender'] = df5['gender'].map( self.target_encode_gender_scaler ).astype(float)

        # region_code - Target Encoding / Frequency Encoding
        df5.loc[:, 'region_code'] = df5['region_code'].map( self.target_encode_region_code_scaler )

        # vehicle_age - One Hot Encoding / Frequency Encoding
        df5 = pd.get_dummies( df5, prefix='vehicle_age', columns=['vehicle_age'] )

        # policy_sales_channel - Target Encoding / Frequency Encoding
        df5.loc[:, 'policy_sales_channel'] = df5['policy_sales_channel'].map( self.fe_policy_sales_channel_scaler )

        col_selected = ['vintage', 'annual_premium', 'age', 'region_code', 'vehicle_damage', 'policy_sales_channel', 'previously_insured']

        return df5[col_selected]
    
    def get_predict(self, model, original_data,test_data):
        #model predict
        pred = model.predict_proba(test_data)

        original_data['score'] = pred

        return original_data.to_json(orient='records', date_format='iso')

    
