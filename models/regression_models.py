from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error


def random_forest_model(x_train, y_train, x_test, y_test):
    """
    This model uses the Random Forest Regression to predict the amount of who finds a review to be helpful.
    :param x_train: a panadaframe that will either hold numerical or vectorized text.
    :param y_train: a pandaframe that holds integers of people who find the reviews to be helpful
    :param x_test: a panadaframe that will either hold numerical or vectorized text.
    :param y_test: a pandaframe that holds integers of people who find the reviews to be helpful
    :return: a list of predictions of how many people they find helpful and the mean absolute error.
    """
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    return pred, mae


def logistic_regression_model(x_train, y_train, x_test, y_test):
    """
    This model uses the Logistic Regression to predict the amount of who finds a review to be helpful.
    :param x_train: a panadaframe that will either hold numerical or vectorized text.
    :param y_train: a pandaframe that holds integers of people who find the reviews to be helpful
    :param x_test: a panadaframe that will either hold numerical or vectorized text.
    :param y_test: a pandaframe that holds integers of people who find the reviews to be helpful
    :return: a list of predictions of how many people they find helpful and the mean absolute error.
    """
    model = LogisticRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    return pred, mae


def svm_regression_model(x_train, y_train, x_test, y_test):
    """
    This model uses the SVM Regression to predict the amount of who finds a review to be helpful.
    :param x_train: a panadaframe that will either hold numerical or vectorized text.
    :param y_train: a pandaframe that holds integers of people who find the reviews to be helpful
    :param x_test: a panadaframe that will either hold numerical or vectorized text.
    :param y_test: a pandaframe that holds integers of people who find the reviews to be helpful
    :return: a list of predictions of how many people they find helpful and the mean absolute error.
    """
    model = svm.SVR()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    return pred, mae


def linear_regression_model(x_train, y_train, x_test, y_test):
    """
    This model uses the Linear Regression to predict the amount of who finds a review to be helpful.
    :param x_train: a panadaframe that will either hold numerical or vectorized text.
    :param y_train: a pandaframe that holds integers of people who find the reviews to be helpful
    :param x_test: a panadaframe that will either hold numerical or vectorized text.
    :param y_test: a pandaframe that holds integers of people who find the reviews to be helpful
    :return: a list of predictions of how many people they find helpful and the mean absolute error.
    """
    model = svm.LinearSVR()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    return pred, mae
