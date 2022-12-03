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
    x_train, x_test, y_train, y_test = bag_of_words(data)

    avg_rf_mae, avg_log_mae, avg_svm_mae, avg_linear_mae = run_models(x_train, x_test, y_train, y_test,
                                                                      number_of_loops)
    print("---------------- Bag-of-Words Results --------------------")
    print("Average MAE for Random Forest Regression = " + str(avg_rf_mae))
    print("Average MAE for Logistic Regression = " + str(avg_log_mae))
    print("Average MAE for SVM Regression = " + str(avg_svm_mae))
    print("Average MAE for Linear Regression = " + str(avg_linear_mae))

    # Now with TD-IDF:
    x_train, x_test, y_train, y_test = tfdif(data)

    avg_rf_mae, avg_log_mae, avg_svm_mae, avg_linear_mae = run_models(x_train, x_test, y_train, y_test,
                                                                      number_of_loops)
    print("----------------- TF-IDF Results --------------------")
    print("Average MAE for Random Forest Regression = " + str(avg_rf_mae))
    print("Average MAE for Logistic Regression = " + str(avg_log_mae))
    print("Average MAE for SVM Regression = " + str(avg_svm_mae))
    print("Average MAE for Linear Regression = " + str(avg_linear_mae))

    # Now lets try with Numerical Values
    x = data.drop(
        ['voteBinary', 'vote', 'verified', 'reviewTime', 'asin', 'style', 'reviewTime', 'reviewText', 'summary',
         'image', 'summary_num_noun', 'summary_num_verb', 'summary_num_adv', 'summary_num_adp',
         'summary_num_propn'], axis=1, inplace=False)
    y = data.loc[:, 'vote']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    avg_rf_mae, avg_log_mae, avg_svm_mae, avg_linear_mae = run_models(x_train, x_test, y_train, y_test,
                                                                      number_of_loops)
    print("---------------- Numerical Results --------------------")
    print("Average MAE for Random Forest Regression = " + str(avg_rf_mae))
    print("Average MAE for Logistic Regression = " + str(avg_log_mae))
    print("Average MAE for SVM Regression = " + str(avg_svm_mae))
    print("Average MAE for Linear Regression = " + str(avg_linear_mae))


    # Demonstration:
    # Find Amazon reviews online, and create demo variables to test out our program.

    # demo_1 = "Aveeno body lotion is our go to moisturizer. Lightly scented and soaks into the skin so oiliness does " \
    #          "not last long but skin feels good. Once we get down to about 1/4 level pump stops picking up. Too much " \
    #          "to waste so cut the container horizontally and put the sizable amount into a Tupperware unit. Lasts " \
    #          "surprisingly long time. Good stuff. "
    # demo_2 = "Two corners are slightly bent, which is a little disappointing, as this is a shower gift."
    # demo_3 = "I read carefully every bad reviews before buying it so I knew that my floors were not gonna be " \
    #          "perfectly clean after using it. After the first cleaning and despite my knowledge that it wasn’t gonna " \
    #          "be perfect, I was deeply disappointed : I did asked myself if it wasn’t dirtier than before. I thought " \
    #          "that iRobot couldn’t have designed a product that bad, so I tried it with washable mopping pads AND I " \
    #          "soaked it with water just before the cleaning. This way, the result is amazing! Highly recommend it! Do " \
    #          "not lose time with the single use mopping pads included. "
    # # Place the demo variables into the array below followed by the number of helpful votes
    # demo_array = [[demo_3, 71], [demo_1, 23], [demo_2, 0]]
    # demo_df = pd.DataFrame(demo_array, columns=['reviewText', 'vote'])
    #
    # print("Before:")
    # print(demo_df)
    # for index, a in enumerate(demo_array):
    #     demo_df.at[index, 'reviewText'] = bag_of_words_demo(demo_df.at[index, 'reviewText'])
    #
    # print("After:")
    # print(demo_df)
    # # We'll start off with Bag-of-Words:
    # x_train, x_test, y_train, y_test = bag_of_words(data)
    #
    # print(x_train)
    # print(demo_df)
    # model = LogisticRegression()
    # model.fit(x_train, y_train)
    # pred = model.predict(demo_df)
    #
    # for index, review in enumerate(demo_array):
    #     print("Review: " + review[0])
    #     print("Predicted 'Helpful Votes Score: " + pred[index])
    #     print("True 'Helpful Votes Score': " + str(review[1]))

