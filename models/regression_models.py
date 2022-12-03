from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error


def random_forest_model(x_train, y_train, x_test, y_test):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    print("Random Forest Regression Training data MAE: " + str(mae))
    return pred, mae

def logistic_regression_model(x_train, y_train, x_test, y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("Logistic Regression Training data MAE: " + str(mean_absolute_error(y_test, pred)))
    return pred


def svm_regression_model(x_train, y_train, x_test, y_test):
    model = svm.SVR()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("SVM Training data MAE: " + str(mean_absolute_error(y_test, pred)))
    return pred


def linear_regression_model(x_train, y_train, x_test, y_test):
    model = svm.LinearSVR()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("Linear Training data MAE: " + str(mean_absolute_error(y_test, pred)))
    return pred