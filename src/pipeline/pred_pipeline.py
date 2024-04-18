import sys
import os
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.utils import load_obj

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict_xyz(self,features) :
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            
            return pred
        except Exception as e:
            logging.info("Error in predict function")





class CustomData:
    def __init__(self,
                 stars:float,
                 reviews: float,
                 boughtInLastMonth : float,
                 category : str,
                 isBestSeller : str
                 ):
        self.stars = stars
        self.category = category
        self.isBestSeller = isBestSeller
        self.reviews = reviews
        self.boughtInLastMonth = boughtInLastMonth


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                       'stars': [self.stars], 
                       'reviews': [self.reviews], 
                       'boughtInLastMonth': [self.boughtInLastMonth], 
                       'category': [self.category],
                       'isBestSeller':[self.isBestSeller]
                  }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe created")
            return df
        except Exception as e:
            logging.info("Error in data as dataframe")