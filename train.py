import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.training_pipeline import DataIngestion, DataTransformation, ModelTrainer

if __name__ == '__main__':
    # Initialize data ingestion
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(f"Training data path: {train_data_path}")
    print(f"Test data path: {test_data_path}")

    # Perform data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    # Train the model
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr) 