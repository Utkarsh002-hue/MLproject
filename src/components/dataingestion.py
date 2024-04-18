import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass




@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Started with data ingestion')
        try:
            df=pd.read_csv(os.path.join('notebooks/data','dataset.csv'))
            logging.info('Data read in dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)


            logging.info('Spliting data into train and test')
            train_split,test_split=train_test_split(df,test_size=0.25,random_state=42)
            train_split.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_split.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingested')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.info('Error occured while data ingestion')
            raise CustomException(e,sys)