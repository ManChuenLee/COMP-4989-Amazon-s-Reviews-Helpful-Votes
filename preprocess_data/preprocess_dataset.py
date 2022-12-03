import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import spacy

TRAINING_DATA_PATH = "AppliancesCsv.csv"
LABEL_NAME = 'vote'

"""
    FEATURES:
reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
asin - ID of the product, e.g. 0000013714
reviewerName - name of the reviewer
helpful - helpfulness rating of the review, e.g. 2/3
reviewText - text of the review
overall - rating of the product
summary - summary of the review
unixReviewTime - time of the review (unix time)
reviewTime - time of the review (raw)
"""


def read_data_from_csv(filepath):
    data = pd.read_csv(filepath, low_memory=False, decimal=',')
    return data


def preprocess_data(data):
    data = replace_nan_with_0(data)
    data = convert_label_to_binary(data)
    data = drop_column(data, 'reviewerName')
    data = drop_column(data, 'reviewerID')
    data = drop_column(data, 'unixReviewTime')
    data = drop_column(data, 'reviewTime')
    data = drop_column(data, 'asin')
    print(data)
    return data


def replace_nan_with_0(data):
    return data.fillna(0)


def convert_label_to_binary(data):
    # replace votes with values > 0 to 1
    data[LABEL_NAME] = pd.to_numeric(data[LABEL_NAME])
    data.loc[(data.vote > 0), LABEL_NAME] = 1
    return data


def add_col(data, col, col_title):
    data[col_title] = col.values
    return data


def drop_column(data, col_name):
    return data.drop(col_name, axis=1, inplace=False)


def dataframe_to_csv(data, name_of_csv):
    data.to_csv(name_of_csv, encoding='utf-8', index=False)


def preprocess_csv1_data():
    dataAppCsv = read_data_from_csv("AppliancesCsv.csv")
    votes = dataAppCsv.loc[:, 'vote']

    data = read_data_from_csv("csv1.csv")
    data = add_col(data, votes, 'vote')
    dataframe_to_csv(data, "csv4.csv")


def test2():
    dataAppCsv = read_data_from_csv("AppliancesCsv.csv")
    dataAppCsv = dataAppCsv.iloc[:10000, :]
    votes = dataAppCsv.loc[:, 'vote']
    print(votes)
    data = read_data_from_csv("csv2.csv")
    data = add_col(data, votes, 'vote')
    dataframe_to_csv(data, "csv3.csv")


def expand_text_feature(data, csv_name):
    tags_bank = ["NOUN", "VERB", "ADJ", "ADV", "ADP", "PROPN"]
    nlp = spacy.load('en_core_web_sm')  # you can use other methods

    # colNames_bank = ["summary_num_noun", "summary_num_verb", "summary_num_adj", "summary_num_adv", "summary_num_adp",
    #                  "summary_num_propn"]

    colNames_bank = ["review_num_noun", "review_num_verb", "review_num_adj", "review_num_adv", "review_num_adp",
                     "review_num_propn"]

    for index in range(4, 6):
        num_tag_items_in_summary = []
        count = 0
        print("\n\n\n\n")
        print(tags_bank[index])
        print("\n\n\n\n")
        for summary in data.reviewText:
            total = 0
            for token in nlp(summary):
                if token.pos_ in tags_bank[index]:
                    total += 1
            num_tag_items_in_summary.append(total)
            count += 1
            if count % 1000 == 0:
                print(count)

        data[colNames_bank[index]] = num_tag_items_in_summary

    data.to_csv(csv_name, encoding='utf-8', index=False)


def expand_review_length(data):
    num_words_review = []
    for review in data.reviewText:
        words = review.split()
        num_words = len(words)
        num_words_review.append(num_words)
    data['review_num_words'] = num_words_review
    print("done")
    return data


def expand_summary_length(data):
    num_words_review = []
    for review in data.summary:
        words = review.split()
        num_words = len(words)
        num_words_review.append(num_words)
    data['summary_num_words'] = num_words_review
    # data.to_csv('csv3_2.csv', encoding='utf-8', index=False)
    print("done")
    return data


def expand_text_length(data):
    data = expand_review_length(data)
    data = expand_summary_length(data)
    data.to_csv('csv3_1.csv', encoding='utf-8', index=False)


def split_dataset(data):
    # split dataset
    training = data.iloc[:70000, :]
    test = data.iloc[70000:, :]
    training.to_csv('training.csv', encoding='utf-8', index=False)
    test.to_csv('test.csv', encoding='utf-8', index=False)


def main():
    pass


if __name__ == '__main__':
    main()
