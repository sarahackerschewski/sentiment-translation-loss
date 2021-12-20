"""

"""
import random
import csv
import numpy as np
from os import listdir
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFpr
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


def preprocess(sequences):
    """

    :return:
    """
    preprocessed_sequences = []
    wn_lem = WordNetLemmatizer()
    for sequence in sequences:
        # remove any HTML tags
        sequence = re.sub(r'<.*?>', '', sequence)
        sequence = re.sub(r'\d+', '', sequence)

        new_seq = ""
        for i, word in enumerate(sequence.split()):
            # stemmer or lemmatizer ?? => so far better results with lemmatizer
            if i == 0:
                new_seq += wn_lem.lemmatize(word)
            else:
                new_seq += " " + wn_lem.lemmatize(word)

        preprocessed_sequences.append(new_seq)

    return preprocessed_sequences

def _replace_abbrevations(word):
    common_abbr = {}
    #if word in common_abbr.keys():


def translate(sentences):
    """

    :param filename:
    :return:
    """
    # translator can only deal with max. 500000 characters
    while sum(len(i) for i in sentences) > 500000:
        random.shuffle(sentences)
        sentences = sentences[:-1]

    return sentences

def sentiment_classifier(x_train, y_train, x_test):
    """

    :return:
    """

    # no stop words removing as they contain important words for sentiment like negation words
    # and thus especially important for a lexical task such as sentiment analysis

    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    tfidf.fit(x_train)
    enc_x_train = tfidf.transform(x_train)
    enc_x_test = tfidf.transform(x_test)
    print(enc_x_train.shape)
    le = LabelEncoder()
    enc_y_train = le.fit_transform(y_train)

    # feature selection
    chi_sq = SelectFpr(chi2, alpha=0.95)
    enc_x_train = chi_sq.fit_transform(enc_x_train, enc_y_train)
    enc_x_test = chi_sq.transform(enc_x_test)
    print(enc_x_train.shape)

    classifier = svm.LinearSVC(C=0.425)
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
        print("Accuracy: ", acc_macro)

        print()

        print("Precision (negative): ", precision_neg)
        print("Recall (negative): ", recall_neg)
        print("F1-Score (negative: ", f1_neg)

        print()

        print("Precision (positive): ", precision_pos)
        print("Recall (positive): ", recall_pos)
        print("F1-Score (positive): ", f1_pos)

    return precision_macro, recall_macro, f1_macro


def create_test():
    # read negative reviews
    neg_files = listdir("../data/English_test/neg/")
    random.shuffle(neg_files)
    neg_files = neg_files
    neg, text = [], []

    for file in neg_files:
        if len(neg) <= 5000:
            file = "../data/English_test/neg/" + file
            print(file)
            content = open(file, "r", encoding="UTF-8").read()
            text.append(content)

            for review in text:
                tokens = sent_tokenize(review)
                for token in tokens:
                    if len(token) > 10:
                        neg.append(token)

    label_neg = ["negative" for i in range(len(neg))]

    # read positive reviews
    pos_files = listdir("../data/English_test/pos/")
    random.shuffle(pos_files)
    pos_files = pos_files
    pos, text = [], []

    for file in pos_files:
        if len(pos) <= 5000:
            file = "../data/English_test/pos/" + file
            print(file)
            content = open(file, "r", encoding="UTF-8").read()
            text.append(content)

            for review in text:
                tokens = sent_tokenize(review)
                for token in tokens:
                    if len(token) > 10:
                        pos.append(token)

    label_pos = ["positive" for i in range(len(pos))]

    test = neg + pos
    gold = label_neg + label_pos
    randomized = list(zip(test, gold))
    random.shuffle(randomized)
    test, gold = zip(*randomized)

    # write file with tokenized English test data
    with open('../data/test_movie_reviews_english_tokenized.csv', 'wt', encoding="UTF-8") as out_file:
        csv_writer = csv.writer(out_file, delimiter=";")
        for i, sent in enumerate(test):
            csv_writer.writerow([gold[i], sent])

    # read German reviews
    test = []
    with open("../data/test_movie_reviews_german.tsv", "r", encoding="UTF-8") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) == 3:
                tokenized = sent_tokenize(row[2], language="german")

                for sent in tokenized:
                    test.append(sent)
    random.shuffle(test)

    # write file with tokenized German test data
    with open('../data/test_movie_reviews_german_tokenized.csv', 'w', encoding="UTF-8") as out_file:
        csv_writer = csv.writer(out_file, delimiter=';')
        for sent in test:
            print(sent)
            csv_writer.writerow([sent])


if __name__ == '__main__':

    # read training data
    train, labels = read_train_data(["../data/training_movie_reviews.txt"])

    # read test data from English movie reviews
    test, gold = [], []
    neg, pos = 0, 0
    with open("../data/test_movie_reviews_english_tokenized.csv") as f:
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
    test = preprocess(test)
    
    # classification
    pred = sentiment_classifier(train, labels, test)

    # evaluation
    evaluate(gold, pred)


