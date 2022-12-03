import string
import re
from keras_preprocessing.text import text_to_word_sequence
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
from nltk.corpus import stopwords


def remove_punctuation(text):
    """
    Removes any punctuations in the text
    :param text: A string that represents the review text.
    :return: A string with punctuations removed.
    """
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


def tokenization(text):
    """
    Tokenizes the text/string.
    :param text: A string that represents the review text.
    :return: An array of strings.
    """
    tokens = re.split('W+', text)
    return tokens


def remove_stopwords(text):
    """
    Removes stopwords in a text/string.
    :param text: A string that represents the review text.
    :return: A string with stopwords removed.
    """
    stopwords_list = stopwords.words('english')
    output = [i for i in text if i not in stopwords_list]
    return output


def lemmatizer(text):
    """
    Lemmatizes the array of text.
    :param text: An array with each element being a single word from the original review Text
    :return: An array of strings.
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text


def preprocess_data(raw_data):
    """
    This function calls all preprocess techniques above to preprocess all the reviewTexts in the dataset.
    :param raw_data: A pandaframe of the raw dataset.
    :return: An pandaframe with a new column that holds preprocessed reviews.
    """
    preprocess_reviews = raw_data
    preprocess_reviews['preprocess_review'] = raw_data['reviewText'].apply(lambda x: remove_punctuation(x))
    preprocess_reviews['preprocess_review'] = preprocess_reviews['preprocess_review'].apply(lambda x: x.lower())
    preprocess_reviews['preprocess_review'] = preprocess_reviews['preprocess_review'].apply(
        lambda x: text_to_word_sequence(x))
    preprocess_reviews['preprocess_review'] = preprocess_reviews['preprocess_review'].apply(
        lambda x: remove_stopwords(x))
    preprocess_reviews['preprocess_review'] = preprocess_reviews['preprocess_review'].apply(lambda x: lemmatizer(x))

    for index, i in enumerate(preprocess_reviews['preprocess_review']):
        if len(i) == 0:
            preprocess_reviews.drop([index], axis=0, inplace=True)
        if len(i) == 1 and len(i[0]) == 1:
            preprocess_reviews.drop([index], axis=0, inplace=True)
        preprocess_reviews['preprocess_review'][index] = " ".join(i)

    return preprocess_reviews
