from argparse import ArgumentParser
import logging
import pandas as pd
import preprocessing
from EDA import EDA


"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""

# implement here your load,preprocess,train,predict,save functions (or any other design you choose)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # 1. load the training set (args.training_set)
    # 2. preprocess the training set
    logging.info("preprocessing train...")
    train = pd.read_csv(args.training_set, encoding="ISO-8859-8")
    proc_features, proc_y_train = preprocessing.preprocess_train(train)

    # plot some data
    eda = EDA(all_data, "try")
    for i in proc_features.columns:
        eda.two_params_correlation(i, "passengers_up")

    # 3. train a model
    logging.info("training...")

    # 4. load the test set (args.test_set)
    # 5. preprocess the test set
    logging.info("preprocessing test...")

    # 6. predict the test set using the trained model
    logging.info("predicting...")

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
