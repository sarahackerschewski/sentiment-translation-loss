"""

"""
import random
import csv
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from nltk import sent_tokenize


def read_train_data(filename_s):
    """
    reads the data from a given file and creates a list of tuples with the polarity and sequence
    :param filename_s: path/name of the file to use;
           defaults to the file "movieReviewSnippets_GroundTruth" from Hutto and Gilbert (2016)
    :return: list of tuples -> (polarity, sequence)
    """
    sequences = []
    labels = []
    for filename in filename_s:
        with open(filename, "r", encoding="UTF-8") as file:
            reader = file.readlines()

            for line in reader:
                line = line.strip("\n")
                line = line.split("\t")

                score = float(line[1])

                if "-" in line[1]:
                    labels.append("negative")
                    sequences.append(line[2])

                elif score != 0.0:
                    labels.append("positive")
                    sequences.append(line[2])

    return (sequences, labels)


def read_test_data_ger(filename):
    """

    :return: the sentence tokenized file content (list)
    """
    text = []
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            text.append(row[2])
    tokenized = sent_tokenize(text)
"""
    # translator can only deal with max. 500000 characters
    while sum(len(i) for i in tokenized) > 500000:
        random.shuffle(tokenized)
        tokenized = tokenized[:-1]

    return tokenized"""


def translate(sentences):
    """

    :param filename:
    :return:
    """


def sentiment_classifier(x_train, y_train, x_test):
    """

    :return:
    """
    """cv = CountVectorizer(ngram_range=(1, 3))
    cv.fit(x_train)

    enc_x_train = cv.transform(x_train)
    enc_x_test = cv.transform(x_test)

    tfidf = TfidfTransformer()
    tfidf.fit(enc_x_train)
    enc_x_train = tfidf.transform(enc_x_train).toarray()
    enc_x_test = tfidf.transform(enc_x_test).toarray()"""
    # no stop words removing as they contain important words for sentiment like negation words
    # and thus especially important for a lexical task such as sentiment analysis
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    tfidf.fit(x_train)
    enc_x_train = tfidf.transform(x_train).toarray()
    enc_x_test = tfidf.transform(x_test).toarray()

    le = LabelEncoder()
    enc_y_train = le.fit_transform(y_train)

    classifier = svm.LinearSVC(C=0.89)
    classifier.fit(enc_x_train, enc_y_train)

    return le.inverse_transform(classifier.predict(enc_x_test))


def evaluate(gold, predicted, verbose=True):
    """

    :param gold:
    :param predicted:
    :param verbose:
    :return:
    """
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(gold, predicted, average="macro")
    acc_macro = accuracy_score(gold, predicted)

    precision_neg, recall_neg, f1_neg, _ = precision_recall_fscore_support(gold, predicted,
                                                                           average="binary", pos_label="negative")
    precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(gold, predicted,
                                                                           average="binary", pos_label="positive")
    if verbose:
        print("Precision (macro-average): ", precision_macro)
        print("Recall (macro-average): ", recall_macro)
        print("F1-Score (macro-average): ", f1_macro)
        print("Accuracy (macro-average): ", acc_macro)

        print()

        print("Precision (negative): ", precision_neg)
        print("Recall (negative): ", recall_neg)
        print("F1-Score (negative: ", f1_neg)

        print()

        print("Precision (positive): ", precision_pos)
        print("Recall (positive): ", recall_pos)
        print("F1-Score (positive): ", f1_pos)

    return precision_macro, recall_macro, f1_macro

def create_test_data():
    neg_files = listdir("../data/English_test_data/neg/")
    random.shuffle(neg_files)
    neg_files = neg_files
    neg, text = [], []

    for file in neg_files:
        if len(neg) <= 5000:
            file = "../data/English_test_data/neg/" + file
            print(file)
            content = open(file, "r", encoding="UTF-8").read()
            text.append(content)

            for review in text:
                tokens = sent_tokenize(review)
                for token in tokens:
                    if len(token) > 10:
                        neg.append(token)

    label_neg = ["negative" for i in range(len(neg))]

    pos_files = listdir("../data/English_test_data/pos/")
    random.shuffle(pos_files)
    pos_files = pos_files
    pos, text = [], []

    for file in pos_files:
        if len(pos) <= 5000:
            file = "../data/English_test_data/pos/" + file
            print(file)
            content = open(file, "r", encoding="UTF-8").read()
            text.append(content)

            for review in text:
                tokens = sent_tokenize(review)
                for token in tokens:
                    if len(token) > 10:
                        pos.append(token)

    label_pos = ["positive" for i in range(len(pos))]

    test_data = neg + pos
    gold = label_neg + label_pos

    randomized = list(zip(test_data, gold))
    random.shuffle(randomized)

    test_data, gold = zip(*randomized)
    with open('../data/test_movie_reviews_english_tokenized.csv', 'wt', encoding="UTF-8") as out_file:
        csv_writer = csv.writer(out_file, delimiter=";")
        for i, sent in enumerate(test_data):
            csv_writer.writerow([gold[i], sent])

    test_data = []
    with open("../data/test_movie_reviews_german.tsv", "r", encoding="UTF-8") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) == 3:
                tokenized = sent_tokenize(row[2], language="german")

                for sent in tokenized:
                    test_data.append(sent)
    random.shuffle(test_data)

    with open('../data/test_movie_reviews_german_tokenized.csv', 'w', encoding="UTF-8") as out_file:
        csv_writer = csv.writer(out_file, delimiter=';')
        for sent in test_data:
            print(sent)
            csv_writer.writerow([sent])

if __name__ == '__main__':

    seq, labels = read_train_data(["../data/training_movie_reviews.txt"])

    # test on English movie reviews

    #pred = sentiment_classifier(seq, labels, test_data)
    #evaluate(gold, pred)


