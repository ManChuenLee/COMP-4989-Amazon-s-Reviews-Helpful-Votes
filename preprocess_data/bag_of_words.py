from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from preprocess_data.preprocess_text import preprocess_data


def bag_of_words(raw_data):
    """
    This function process the review texts via Bag-of_Words
    :param raw_data: The raw data from the dataset
    :return: The x_train, x_test, y_train, y_test of the data.
    """
    # new_data = preprocess_data(raw_data)
    vectorizer = CountVectorizer(analyzer='word')

    x_train, x_test, y_train, y_test = train_test_split(raw_data['reviewText'], raw_data["vote"], test_size=0.3,
                                                        random_state=42)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    return x_train, x_test, y_train, y_test
