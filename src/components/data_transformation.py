import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols =     ['workclass', 'education', 'marital_status', 'occupation',       'relationship', 'race', 'sex', 'native_country']#, 'Class']

            numerical_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',     'hours_per_week']
            
       
               
            # Define the custom ranking for each ordinal variable
            workclass_cat = [' State-gov',' Self-emp-not-inc',' Private',' Federal-gov',' Local-gov', ' Self-emp-inc',' Without-pay',' Never-worked']
            education_cat = [' Bachelors',' HS-grad',' 11th',' Masters',' 9th',' Some-college',' Assoc-acdm',' Assoc-voc',' 7th-8th',' Doctorate',' Prof-school' ,' 5th-6th',' 10th',' 1st-4th',' Preschool',' 12th']
            marital_status_cat = [' Never-married',' Married-civ-spouse',' Divorced', ' Married-spouse-absent',' Separated',' Married-AF-spouse',' Widowed']
            occupation_cat = [' Adm-clerical',' Exec-managerial',' Handlers-cleaners',' Prof-specialty', ' Other-service',' Sales',' Craft-repair',' Transport-moving', ' Farming-fishing',' Machine-op-inspct',' Tech-support', ' Protective-serv',' Armed-Forces',' Priv-house-serv']
            relationship_cat = [' Not-in-family',' Husband',' Wife',' Own-child',' Unmarried', ' Other-relative']
            race_cat = [' White',' Black',' Asian-Pac-Islander',' Amer-Indian-Eskimo',' Other']
            sex_cat = [' Male',' Female']
            native_country_cat = [' United-States',' Cuba',' Jamaica',' India',' ?',' Mexico',' South', ' Puerto-Rico',' Honduras',' England',' Canada',' Germany',' Iran', ' Philippines',' Italy',' Poland',' Columbia',' Cambodia',' Thailand', ' Ecuador',' Laos',' Taiwan',' Haiti',' Portugal',' Dominican-Republic', ' El-Salvador',' France',' Guatemala',' China',' Japan',' Yugoslavia', ' Peru',' Outlying-US(Guam-USVI-etc)',' Scotland',' Trinadad&Tobago', ' Greece',' Nicaragua',' Vietnam',' Hong',' Ireland',' Hungary', ' Holand-Netherlands']
            #Class = [40000 60000]
        

      
           

            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            logging.info('Pipeline Initiated 11')
            # Categorigal Pipeline
                     
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                #('ordinalencoder',OrdinalEncoder(categories=[weather_cat,Road_traffic_density_cat,Type_of_order_cat,Type_of_vehicle_cat,Festival_cat,City_cat])),
                #('ordinalencoder',OrdinalEncoder(categories=[workclass_cat,education_cat ,marital_status_cat,occupation_cat,relationship_cat, race_cat,sex_cat,native_country_cat,class_cat])),
                ('ordinalencoder',OrdinalEncoder(categories=[workclass_cat,education_cat ,marital_status_cat,occupation_cat,relationship_cat, race_cat,sex_cat,native_country_cat])),
               #  ('label_encoder', LabelEncoder()),
                 ('scaler',StandardScaler())
                ]

            )
            logging.info('Pipeline Initiated 22')
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)


          
            ])
            logging.info('Pipeline Initiated 333333333333333')
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        logging.info('ini data transformation start 00')
        try:
            # Reading train and test data
            logging.info('ini data transformation start ')
            train_df = pd.read_csv(train_path)
            logging.info('ini data transformation start 1')
            test_df = pd.read_csv(test_path)
            logging.info('ini data transformation start 2')
            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()
            logging.info('ini data transformation start 3')
            target_column_name = 'Class'#'price'cls

            drop_columns = [target_column_name]#,'id']
            logging.info('ini data transformation start 4')
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            logging.info('ini data transformation start 5')
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            print('ini data transformation start 6')
            target_feature_test_df=test_df[target_column_name]
            print('ini data transformation start 7')
            print('TRAIN###################\n',input_feature_train_df.sample(5))
            print('TEST########',input_feature_test_df.sample(5))
          

           


            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            print('ini data transformation start 8')
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            print('ini data transformation start 9')
            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)