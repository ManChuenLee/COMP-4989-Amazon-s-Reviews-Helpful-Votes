import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import numpy as np
from sklearn.model_selection import KFold
import pickle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

DATASET_PATH = '../datasets/preprocessed_appliance_amazon.csv'

# ###############################################################3
#
#
# ! This classification code has been archived !
#   is not used in final version of project ...
# ! This classification code has been archived !
#
#
# ###############################################################3

def get_data(filepath):
    data = pd.read_csv(filepath, low_memory=False, decimal=',')
    data = drop_column(data, 'reviewTime')
    data = drop_column(data, 'asin')
    data = drop_column(data, 'style')
    data = drop_column(data, 'vote')
    data = drop_column(data, 'reviewText')
    data = drop_column(data, 'summary')
    return data


def drop_column(data, col_name):
    return data.drop(col_name, axis=1, inplace=False)


def random_forest(train_features, train_labels, n_estimators=120, max_depth=40, **kwargs):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, **kwargs)
    model.fit(train_features, train_labels)
    return model


def split_train_test(data):
    # split the data into training set and test set
    # use 75 percent of the data to train the model and hold back 25 percent
    # for testing
    train_ratio = 0.75
    # number of samples in the data_subset
    num_rows = data.shape[0]
    # shuffle the indices
    shuffled_indices = list(range(num_rows))
    random.seed(42)
    random.shuffle(shuffled_indices)

    # calculate the number of rows for training
    train_set_size = int(num_rows * train_ratio)

    # training set: take the first 'train_set_size' rows
    train_indices = shuffled_indices[:train_set_size]
    # test set: take the remaining rows
    test_indices = shuffled_indices[train_set_size:]

    # create training set and test set
    train_data = data.iloc[train_indices, :]
    test_data = data.iloc[test_indices, :]

    # prepare training features and training labels
    # features: all columns except 'y'
    # labels: 'y' column
    train_features = train_data.drop('voteBinary', axis=1, inplace=False)
    train_labels = train_data.loc[:, 'voteBinary']

    # prepare test features and test labels
    test_features = test_data.drop('voteBinary', axis=1, inplace=False)
    test_labels = test_data.loc[:, 'voteBinary']

    return (train_features, train_labels), (test_features, test_labels)


def split_train_test_2(features, labels):
    features['voteBinary'] = labels.values
    data = features

    # split the data into training set and test set
    # use 75 percent of the data to train the model and hold back 25 percent
    # for testing
    train_ratio = 0.75
    # number of samples in the data_subset
    num_rows = data.shape[0]
    # shuffle the indices
    shuffled_indices = list(range(num_rows))
    random.seed(42)
    random.shuffle(shuffled_indices)

    # calculate the number of rows for training
    train_set_size = int(num_rows * train_ratio)

    # training set: take the first 'train_set_size' rows
    train_indices = shuffled_indices[:train_set_size]
    # test set: take the remaining rows
    test_indices = shuffled_indices[train_set_size:]

    # create training set and test set
    train_data = data.iloc[train_indices, :]
    test_data = data.iloc[test_indices, :]

    # prepare training features and training labels
    # features: all columns except 'y'
    # labels: 'y' column
    train_features = train_data.drop('voteBinary', axis=1, inplace=False)
    train_labels = train_data.loc[:, 'voteBinary']

    # prepare test features and test labels
    test_features = test_data.drop('voteBinary', axis=1, inplace=False)
    test_labels = test_data.loc[:, 'voteBinary']

    return (train_features, train_labels), (test_features, test_labels)


def test():
    # get training and test data
    data = get_data('training.csv')
    check_imbalance_of_label(data)
    data = balance_dataset(data)
    check_imbalance_of_label(data)


def balance_dataset(data):
    num_positive = (data['voteBinary'] == 1).sum()
    num_negative = (data['voteBinary'] == 0).sum()
    total = len(data)
    diff_positive_negative = abs(num_negative - num_positive)

    # split dataset
    x = num_positive + diff_positive_negative
    filler_data = get_data('csv1_6.csv')
    df_2 = filler_data.iloc[num_positive:x, :]

    balanced_dataset = pd.concat([data, df_2])
    return balanced_dataset


def main():
    # get training and test data
    data = get_data(DATASET_PATH)

    # address imbalanced data
    data = balance_dataset(data)

    #   Random under-sampling
    # rus = RandomUnderSampler(sampling_strategy=1)  # 1 meaning a 1:1 ratio
    # features, labels = rus.fit_resample(features, labels)
    # print(len(features))

    #   Random over-sampling
    # ros = RandomOverSampler(sampling_strategy=1)  # 1 meaning a 1:1 ratio
    # features, labels = ros.fit_resample(features, labels)
    # print(len(features))

    training_set, test_set = split_train_test(data)

    train_features = training_set[0]
    train_labels = training_set[1]

    print(train_features)
    print(train_labels)

    #   test knn classification
    # neigh = KNeighborsClassifier(n_neighbors=10)
    # neigh.fit(train_features, train_labels)
    # predictions = neigh.predict(test_features)
    # my_f1_score = f1_score(test_labels, predictions)
    # print(my_f1_score)

    #   test rf classification
    # kwargs = {"n_estimators": 100, "max_depth": 60}
    # rf_model = random_forest(train_features, train_labels, **kwargs)
    # predictions = rf_model.predict(test_features)
    #
    # my_f1_score = f1_score(test_labels, predictions)
    # print(my_f1_score)

    evaluate_classification_model(train_features, train_labels)


def check_imbalance_of_label(data):
    num_positive = (data['voteBinary'] == 1).sum()
    num_negative = (data['voteBinary'] == 0).sum()
    total = len(data)
    print("Number of accepted reviews == %d" % num_positive)
    print("Number of rejected reviews == %d" % num_negative)
    print("# of reviews accepted == {:.4f}%".format(num_positive / total))
    print("# of reviews rejected == {:.4f}%".format(num_negative / total))


def export_classification_model(train_features, train_label):
    model = random_forest(train_features, train_label)
    pickle.dump(model, open('classification_model.pkl', 'wb'))


def evaluate_classification_model(train_features, train_label):
    sample_cv_errors = []
    sample_train_errors = []

    kf = KFold(n_splits=5)
    kf.get_n_splits(train_features)

    my_kwargs = [
        {"n_estimators": 120, "max_depth": 40},
        {"n_estimators": 120, "max_depth": 40},
    ]

    for kwarg in my_kwargs:
        cv_errors = []
        train_errors = []

        for train_index, test_index in kf.split(train_features):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = train_features.iloc[train_index], train_features.iloc[test_index]
            y_train, y_test = train_label.iloc[train_index], train_label.iloc[test_index]

            print("rf model using:")
            for kw in kwarg:
                print("\t%s : %s" % (kw, kwarg[kw]))
            model = random_forest(X_train, y_train, **kwarg)

            valid_pred = model.predict(X_test)
            cv_errors.append(f1_score(y_test, valid_pred))

            #   USE confusion matrix to evaluate model
            # TN, FP, FN, TP = confusion_matrix(y_test.to_numpy(), valid_pred).ravel()
            # print("True positive(TP) = ", TP)
            # print("False positive(FP) = ", FP)
            # print("True Negative(TN) = ", TN)
            # print("False Negative(FN) = ", FN)
            # accuracy = (TP + TN) / (TP + TN + FP + FN)
            # print("Accuracy = " + str(accuracy))

            train_pred = model.predict(X_train)
            train_errors.append(f1_score(y_train, train_pred))

        print("avg cv error: ", np.average(cv_errors))
        print("avg train error: ", np.average(train_errors))
        print()
        sample_train_errors.append(np.average(train_errors))
        sample_cv_errors.append(np.average(cv_errors))


if __name__ == '__main__':
    main()


