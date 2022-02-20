"""
Author: Sarah Ackerschewski
Bachelor Thesis
WS 21/22
Supervisor: Cagri Cöltekin

This file contains the code for building a sentiment analyzer (LinearSVM).
It also contains functions for reading the training data, preprocessing the data
and evaluating the predictions by the SVM,
as well as a function for applying the TextBlob sentiment lexicon to data.
"""

import csv
import re
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFpr
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from textblob import TextBlob


def read_train_data(filename_s):
    """
    reads the data from a given file and creates a list of tuples with the polarity and sequence
    :param filename_s: path/name of the file to use
    :return: tuple of lists -> (sequences, polarity)
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


def preprocess(sequences):
    """
    preprocesses the sentences by lemmatizing the sentences and removing HTML tags
    :param sequences: the sentences to preprocess
    :return: preprocessed sequences
    """
    preprocessed_sequences = []
    wn_lem = WordNetLemmatizer()
    for sequence in sequences:
        # remove any HTML tags
        sequence = sequence.lower()
        sequence = re.sub(r'<.*?>', '', sequence)

        seq_list = sequence.split()
        seq_list = list(map(lambda word: wn_lem.lemmatize(word), seq_list))
        sequence = ' '.join(seq_list)

        preprocessed_sequences.append(sequence)

    return preprocessed_sequences


def sentiment_classifier(x_train, y_train, x_test, C=1.0, save_pred=False, y_test=None, filename=""):
    """
    Data Transformation: creates feature vectors for the data; performs Chi-squared feature selection
    Model: builds a LinearSVM, validated with 10-fold cross-validation and fitted with transformed training data
    :param x_train: training sentences
    :param y_train: labels for the training sentences
    :param x_test: test sentences
    :param C: tuning parameter; Default=1.0
    :return: polarity predictions for the test data, cross-validation results (were needed for tuning)
    """
    # create TFDIF feature vectors
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    tfidf.fit(x_train)
    enc_x_train = tfidf.transform(x_train)
    enc_x_test = tfidf.transform(x_test)
    le = LabelEncoder()
    enc_y_train = le.fit_transform(y_train)

    # feature selection
    chi_sq = SelectFpr(chi2, alpha=0.9)
    enc_x_train = chi_sq.fit_transform(enc_x_train, enc_y_train)
    enc_x_test = chi_sq.transform(enc_x_test)

    # build sentiment classifier
    classifier = svm.LinearSVC(C=C, max_iter=10000)

    # cross-validation
    scores = cross_val_score(classifier, enc_x_train, enc_y_train, cv=10, scoring="accuracy")

    print("10-fold cross-validation:")
    print("Accuracy: %0.2f; SD: %0.2f" % (scores.mean(), scores.std()))

    classifier.fit(enc_x_train, enc_y_train)

    predictions, tpl = le.inverse_transform(classifier.predict(enc_x_test)), (scores.mean(), scores.std())

    # writes predictions, gold label and sentence to csv file
    if save_pred:
        path = "../results/results_" + filename + ".csv"
        with open(path, "wt", encoding="UTF-8") as out_f:
            csv_writer = csv.writer(out_f, delimiter=";")
            for i, pred in enumerate(predictions):
                csv_writer.writerow([pred, y_test[i], x_test[i]])

    return predictions, tpl


def evaluate(gold, predicted, verbose=True):
    """
    calculates macro-averaged precision, recall and F1-measure for the predictions
    :param gold: "ideal" labels of the data
    :param predicted: polarity predicitions by the
    :param verbose: prints results to Console if True; Default: True
    :return: evaluation results of each measure
    """
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(gold, predicted, average="macro")
    acc_macro = accuracy_score(gold, predicted)

    if verbose:
        print("Precision (macro-average): ", precision_macro)
        print("Recall (macro-average): ", recall_macro)
        print("F1-Score (macro-average): ", f1_macro)
        print("Accuracy: ", acc_macro)

    return precision_macro, recall_macro, f1_macro, acc_macro



def lexicon_based_analyzer_eng(data):
    """
    "predicts" the polarity for the given data with the TextBlob lexicon
    :param data: sentences to classify as positive or negative
    :return: classified sentences
    """
    pred = []

    for sent in data:
        sentiment = TextBlob(sent).sentiment.polarity
        if sentiment < 1:
            pred.append("positive")
        elif sentiment > 0:
            pred.append("negative")
        else:
            pred.append("_")
    return pred


if __name__ == '__main__':
    # read training data
    train, labels = read_train_data(["../data/training_movie_reviews.txt"])

    # read test data from English movie reviews
    test, gold = [], []
    neg, pos = 0, 0
    with open("../data/test_movie_reviews_english_tokenized.csv", "r", encoding="UTF-8") as f:
        csv_reader = csv.reader(f, delimiter=";")

        for row in csv_reader:
            if len(row) == 2:
                if row[0] == "negative" and neg < 1000:
                    neg += 1
                    gold.append(row[0])
                    test.append(row[1])

                elif row[0] == "positive" and pos < 1000:
                    pos += 1
                    gold.append(row[0])
                    test.append(row[1])

    # pre-processing
    train = preprocess(train)
    preprocessed_test = preprocess(test)

    print("English data:")

    C = 375

    pred, tpl = sentiment_classifier(train, labels, preprocessed_test, C=C)

    print("\nEvaluation SVM")
    _, _, _, accuracy = evaluate(gold, pred, verbose=True)

    print("\nEvaluation Lexicon-based")
    lexicon_pred = lexicon_based_analyzer_eng(test)
    _, _, _, acc = evaluate(gold, lexicon_pred, verbose=True)

    # test DeepL translations

    test2, gold2 = [], []
    with open("../data/translated_german_test_data.csv", "r", encoding="UTF-8") as t_file:
        csv_reader = csv.reader(t_file, delimiter=";")

        for row in csv_reader:
            if len(row) == 2:
                test2.append(row[1])
                gold2.append(row[0])

    preprocessed_test2 = preprocess(test2)

    print("\nGerman data:")
    print("\n-----DeepL translations-----")
    C = 375
    pred2,_ = sentiment_classifier(train, labels, preprocessed_test2, C=C)
    print(len(pred2))
    print("\nEvaluation SVM")
    _, _, _, _ = evaluate(gold2, pred2, verbose=True)

    print("\nEvaluation Lexicon-based")
    lexicon_pred2 = lexicon_based_analyzer_eng(test2)
    _, _, _, _ = evaluate(gold2, lexicon_pred2, verbose=True)

    # test Google translations

    print("\n-----Transformer translations-----")

    test3, gold3 = [], []
    with open("../data/other_translated_german_test_data.csv", "r", encoding="UTF-8") as t2_file:
        csv_reader = csv.reader(t2_file, delimiter=";")
        pos, neg = 0,0
        for row in csv_reader:
            if len(row) == 2:
                if row[0] == "negative" and neg < 1000:
                    neg += 1
                    gold3.append(row[0])
                    test3.append(row[1])

                elif row[0] == "positive" and pos < 1000:
                    pos += 1
                    gold3.append(row[0])
                    test3.append(row[1])

    preprocessed_test3 = preprocess(test3)

    C = 375

    pred3,_ = sentiment_classifier(train, labels, preprocessed_test3, C=C, save_pred=True, y_test=gold3, filename="de-en_other")
    print("\nEvaluation SVM")
    _, _, _, _ = evaluate(gold3, pred3, verbose=True)

    print("\nEvaluation Lexicon-based")
    lexicon_pred3 = lexicon_based_analyzer_eng(test3)
    _, _, _, _ = evaluate(gold3, lexicon_pred3, verbose=True)

    print("\nOther domain:")
    
    # test other domain => hotel review
    test4 = []
    gold4 = []
    len_pos, len_neg = 0, 0
    with open("../data/test_hotel_reviews_tokenized.csv") as f:
        csv_reader = csv.reader(f, delimiter=";")

        for row in csv_reader:
            if len(row) == 2:
                if row[0] == "negative" and len_neg < 1000:
                    len_neg += 1
                    gold4.append(row[0])
                    test4.append(row[1])

                elif row[0] == "positive" and len_pos < 1000:
                    len_pos += 1
                    gold4.append(row[0])
                    test4.append(row[1])

    preprocessed_test4 = preprocess(test4)
    C = 375
    pred4, _ = sentiment_classifier(train, labels, preprocessed_test4, C=C)
    print("\nEvaluation SVM")
    _, _, _, _ = evaluate(gold4, pred4, verbose=True)

    print("\nEvaluation Lexicon-based")
    lexicon_pred4 = lexicon_based_analyzer_eng(test4)
    _, _, _, _ = evaluate(gold4, lexicon_pred4, verbose=True)


