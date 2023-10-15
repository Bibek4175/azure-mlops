# Import libraries
import argparse
import glob
import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlflow.sklearn import autolog

# Define functions


def main(args):
    # Enable autologging
    autolog()

    # Read data
    df = get_csvs_df(args.training_data)

    # Split data and train model
    X_train, X_test, y_train, y_test = split_data(df)
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use a non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in the provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# Our new function
def split_data(df, test_size=0.2):
    X = df.drop("Diabetic", axis=1)
    y = df["Diabetic"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # Train model
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X_train, y_train)


def parse_args():
    # Setup arg parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--training_data", dest="training_data", type=str)
    parser.add_argument("--reg_rate", dest="reg_rate", type=float, default=0.01)

    # Parse args
    args = parser.parse_args()

    # Return args
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
