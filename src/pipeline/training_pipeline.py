import os
import sys
from src.logger.logging import logging
from src.exceptions.exception import customexception
import pandas as pd

from components.data_iingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


data_ingestion=DataIngestion()

train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

data_transformation=DataTransformation()

train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)


model_trainer_obj=ModelTrainer()
model_trainer_obj.initiate_model_training(train_arr,test_arr)

# model_eval_obj = ModelEvaluation()
# model_eval_obj.initiate_model_evaluation(train_arr,test_arr)