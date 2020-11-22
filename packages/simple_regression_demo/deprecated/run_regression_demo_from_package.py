from simple_regression_demo.processing.local_data_management import create_regression_demo_data, make_splits, load_dataset
from simple_regression_demo.train_pipeline import run_training
from simple_regression_demo.predict import make_prediction

create_regression_demo_data(mode="production")
create_regression_demo_data(mode="testing")

make_splits()
make_splits(mode="testing")

run_training()

input_data = load_dataset(mode="testing", file_name="test.csv")
make_prediction(input_data=input_data)
input_data = load_dataset(mode="production", file_name="test.csv")
make_prediction(input_data=input_data)