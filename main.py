import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from models.regression_models import random_forest_model, logistic_regression_model, svm_regression_model, \
    linear_regression_model
from preprocess_data.bag_of_words import bag_of_words, bag_of_words_demo
from preprocess_data.tf_idf import tfdif


def average(lst):
    """
    Find the average within a list of numerical values
    :param lst: a list of numerical values
    :return: an int that represents the MAE score of a model
    """
    return sum(lst) / len(lst)


def run_models(new_x_train, new_x_test, new_y_train, new_y_test, new_number_of_loops):
    """
    This function runs all the regression models.
    :param new_x_train: A dataframe of reviews samples to be trained on
    :param new_x_test: A dataframe of review samples to be tested on
    :param new_y_train: A dataframe of votes to be trained on
    :param new_y_test: A dataframe of votes to be test on
    :param new_number_of_loops: Number of times the for loop will run
    :return: A tuple of average MAE scores of corresponding models.
    """
    rf_mae_list = []
    log_mae_list = []
    svm_mae_list = []
    linear_mae_list = []

    for x in range(new_number_of_loops):
        print("Running loop: " + str(x+1))
        svm_pred, svm_mae = svm_regression_model(new_x_train, new_y_train, new_x_test, new_y_test)
        linear_pred, linear_mae = linear_regression_model(new_x_train, new_y_train, new_x_test, new_y_test)
        rf_pred, rf_mae = random_forest_model(new_x_train, new_y_train, new_x_test, new_y_test)
        log_pred, log_mae = logistic_regression_model(new_x_train, new_y_train, new_x_test, new_y_test)
        rf_mae_list.append(rf_mae)
        log_mae_list.append(log_mae)
        svm_mae_list.append(svm_mae)
        linear_mae_list.append(linear_mae)
        print("Finished loop: " + str(x+1))

    new_avg_rf_mae = average(rf_mae_list)
    new_avg_log_mae = average(log_mae_list)
    new_avg_svm_mae = average(svm_mae_list)
    new_avg_linear_mae = average(linear_mae_list)

    return new_avg_rf_mae, new_avg_log_mae, new_avg_svm_mae, new_avg_linear_mae


if __name__ == '__main__':
    # This is where we read the csv file to grab the data
    data = pd.read_csv("./datasets/preprocessed_appliance_amazon.csv", low_memory=False)[0:2000]

    number_of_loops = 5

    # We'll start off with Bag-of-Words:
    # x_train, x_test, y_train, y_test = bag_of_words(data)
    #
    # avg_rf_mae, avg_log_mae, avg_svm_mae, avg_linear_mae = run_models(x_train, x_test, y_train, y_test,
    #                                                                   number_of_loops)
    # print("---------------- Bag-of-Words Results --------------------")
    # print("Average MAE for Random Forest Regression = " + str(avg_rf_mae))
    # print("Average MAE for Logistic Regression = " + str(avg_log_mae))
    # print("Average MAE for SVM Regression = " + str(avg_svm_mae))
    # print("Average MAE for Linear Regression = " + str(avg_linear_mae))
    #
    # # Now with TD-IDF:
    # x_train, x_test, y_train, y_test = tfdif(data)
    #
    # avg_rf_mae, avg_log_mae, avg_svm_mae, avg_linear_mae = run_models(x_train, x_test, y_train, y_test,
    #                                                                   number_of_loops)
    # print("----------------- TF-IDF Results --------------------")
    # print("Average MAE for Random Forest Regression = " + str(avg_rf_mae))
    # print("Average MAE for Logistic Regression = " + str(avg_log_mae))
    # print("Average MAE for SVM Regression = " + str(avg_svm_mae))
    # print("Average MAE for Linear Regression = " + str(avg_linear_mae))
    #
    # # Now lets try with Numerical Values
    # x = data.drop(
    #     ['voteBinary', 'vote', 'verified', 'reviewTime', 'asin', 'style', 'reviewTime', 'reviewText', 'summary',
    #      'image', 'summary_num_noun', 'summary_num_verb', 'summary_num_adv', 'summary_num_adp',
    #      'summary_num_propn'], axis=1, inplace=False)
    # y = data.loc[:, 'vote']
    #
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    #
    # avg_rf_mae, avg_log_mae, avg_svm_mae, avg_linear_mae = run_models(x_train, x_test, y_train, y_test,
    #                                                                   number_of_loops)
    # print("---------------- Numerical Results --------------------")
    # print("Average MAE for Random Forest Regression = " + str(avg_rf_mae))
    # print("Average MAE for Logistic Regression = " + str(avg_log_mae))
    # print("Average MAE for SVM Regression = " + str(avg_svm_mae))
    # print("Average MAE for Linear Regression = " + str(avg_linear_mae))


    # Demonstration:
    # Find Amazon reviews online, and create demo variables to test out our program.
    demo_data = pd.read_csv("./datasets/sample2_demo.csv", low_memory=False)

    x_train, x_test, y_train, y_test = bag_of_words(data)
    x_train_demo, x_test_demo = bag_of_words_demo(demo_data)

    print(x_train)
    print(x_train_demo)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    print("x_train.shape()")
    print(np.shape(x_train))
    print("x_train_demo.shape()")
    print(np.shape(x_train_demo))

    np.reshape(x_train_demo, (-1, 6031))
    pred = model.predict(x_train_demo)

