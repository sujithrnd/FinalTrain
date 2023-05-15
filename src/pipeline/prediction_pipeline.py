import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object 
import pandas as pd


class Predictionpipeline:
    def __ini__(self):
        pass
    def predict(self,features):
        try:
            logging.info('start predict ')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            logging.info('Load preprocessor.pkl  with preprocessor_path=',preprocessor_path)
            preprocessor=load_object(preprocessor_path)
            logging.info('Load model.pkl  with model_path=',model_path)
            model=load_object(model_path)
            logging.info('transform feautures')
            data_scaled=preprocessor.transform(features)
            logging.info('Predict')
            pred=model.predict(data_scaled)
            logging.info('Return prediction ')
            return pred
        except Exception as e:
            logging.info('Exception occured in Prediction')
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(seld,                 
                weather         :str, 
                road_traffic    :str,
                type_order      :str ,
                type_vehicle    :str,
                festival        :str ,
                city            :str,

                Delivery_person_Age     :int,
                Delivery_person_Ratings   :int,
                Vehicle_condition           :int,
                multiple_deliveries            :int,
                Day:int,
                Month:int,
                Year:int,
                Order_Hr:int,
                Order_Min:int,
                Order_Sec:int):


                self.weather         :weather
                self.road_traffic    :road_traffic
                self.type_order      :type_order 
                self.type_vehicle    :type_vehicle
                self.festival        :festival 
                self.city            :city

                self.Delivery_person_Age     :Delivery_person_Age
                self.Delivery_person_Ratings   :Delivery_person_Ratings
                self.Vehicle_condition           :Vehicle_condition
                self.multiple_deliveries            :multiple_deliveries
                self.Day:Day
                self.Month:Month
                self.Year:Year
                self.Order_Hr:Order_Hr
                self.Order_Min:Order_Min
                self.Order_Sec:Order_Sec
           




    """
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:float,
                 color:str,
                 clarity:str):
     
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut=cut
        self.color=color
        self.clarity=clarity"""

    def  get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'weather':[self.weather],
                'road_traffic':[self.road_traffic],
                'type_order':[self.type_order],
                 'type_vehicle':[self.type_vehicle],
                 'festival':[self.festival],
                 'city':[self.city],
                  'Delivery_person_Age':[self.Delivery_person_Age],
                  'Delivery_person_Ratings':[self.Delivery_person_Ratings],

                  'Vehicle_condition':[self.Vehicle_condition],
                  'multiple_deliveries':[self.multiple_deliveries],
                  'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                  'Day':[self.Day],
                  'Month':[self.Month],
                  'Year':[self.Year],
                  'Order_Hr':[self.Order_Hr],
                  'Order_Min':[self.Order_Min],
                  'Order_Sec':[self.Order_Sec],


            }
            custom_data_input_dict11 = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                 'x':[self.x],
                 'y':[self.y],
                 'cut':[self.cut],
                  'color':[self.color],
                  'clarity':[self.clarity]
            }
            
            }	
            df=pd.DataFrame(custom_data_input_dict)
            logging.info('Data Frame gathered')
            return df
        except Exception as e:
            logging.info('Exception occured prediction pipeline')
            raise CustomException(e,sys)
            
            
            
            
            



