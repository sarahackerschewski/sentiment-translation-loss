"""

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
import test_data_creation

abbr_slang_dict = {"awsm": "awesome",
                   "adorbs": "adorable",
                   "asap": "as soon as possible",
                   "bc": "because",
                   "btw": "by the way",
                   "bts": "behind the scenes",
                   "cmv": "change my view",
                   "dae": "does anyone else",
                   "eom": "end of message",
                   "eot": "end of thread",
                   "fwiw": "for what it is worth",
                   "fyi": "for your information",
                   "ftw": "for the win",
                   "hes": "he is",
                   "hth": "hope this helps",
                   "icymi": "in case you missed it",
                   "idc": "i do not care",
                   "idk": "i do not know",
                   "im": "i am",
                   "imo": "in my opinion",
                   "mho": "in my humble opinion",
                   "irl": "in real life",
                   "iirc": "if i remember correctly",
                   "jk": "just kidding",
                   "lmk": "let me know",
                   "l8r": "later",
                   "lmao": "laughing my ass off",
                   "lol": "laughing out loud",
                   "luv": "love",
                   "nm": "nevermind",
                   "nvm": "nevermind",
                   "noob": "newcomer",
                   "ooak": "one of a kind",
                   "ofc": "of course",
                   "ok": "okay",
                   "omg": "oh my god",
                   "otoh": "on the other hand",
                   "rn": "right now",
                   "shes": "she is",
                   "smh": "shaking my head",
                   "tbh": "to be honest",
                   "theyre": "they are",
                   "til": "today i learned",
                   "tmi": "too much information",
                   "totes": "totally",
                   "u": "you",
                   "w/": "with",
                   "w/o": "without",
                   "wth": "what the hell",
                   "wtf": "what the fuck",
                   "y": "why",
                   "youre": "you are",
                   "yw": "you are welcome",
                   "ywa": "you are welcome anyway",
                   "&": "and"}


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
        sequence = sequence.lower()
        sequence = re.sub(r'<.*?>', '', sequence)

        # no removing slang ==> can indicate sentiment
        """seq_list = sequence.split()
        seq_list = list(map(lambda word: _replace_abbrevations(word), seq_list))
        sequence = ' '.join(seq_list)"""

        # no removing non-alphabetic char ==> e.g. ASCII emojis also convey sentiment
        # sequence = re.sub(r'[^a-züäöß ]*', '', sequence)

        seq_list = sequence.split()
        seq_list = list(map(lambda word: wn_lem.lemmatize(word), seq_list))
        sequence = ' '.join(seq_list)

        preprocessed_sequences.append(sequence)

    return preprocessed_sequences


def _replace_abbrevations(word):
    if word in abbr_slang_dict.keys():
        return abbr_slang_dict[word]
    return word


def sentiment_classifier(x_train, y_train, x_test, C=1.0):
    """

    :return:
    """

    # no stop words removing as they contain important words for sentiment like negation words
    # and thus especially important for a lexical task such as sentiment analysis

    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    tfidf.fit(x_train)
    enc_x_train = tfidf.transform(x_train)
    enc_x_test = tfidf.transform(x_test)
    le = LabelEncoder()
    enc_y_train = le.fit_transform(y_train)

    # feature selection
    chi_sq = SelectFpr(chi2, alpha=0.85)
    enc_x_train = chi_sq.fit_transform(enc_x_train, enc_y_train)
    enc_x_test = chi_sq.transform(enc_x_test)

    classifier = svm.LinearSVC(C=C)
    # cross-validation
    scores = cross_val_score(classifier, enc_x_train, enc_y_train, cv=10, scoring="accuracy")

    print("10-fold cross-validation:")
    print("Accuracy: %0.2f; SD: %0.2f" % (scores.mean(), scores.std()))

    classifier.fit(enc_x_train, enc_y_train)

    return le.inverse_transform(classifier.predict(enc_x_test)), (scores.mean(), scores.std())


def lexicon_based_analyzer_eng(data):
    pred = []
    for sent in data:
        seq_list = list(map(lambda word: _replace_abbrevations(word), sent.split()))
        sent = ' '.join(seq_list)
        sentiment = TextBlob(sent).sentiment.polarity
        if sentiment < 1:
            pred.append("positive")
        elif sentiment > 0:
            pred.append("negative")
        else:
            pred.append("_")
    return pred


def evaluate(gold, predicted, verbose=True):
    """

    :param gold:
    :param predicted:
    :param verbose:
    :return:
    """
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(gold, predicted, average="macro")
    acc_macro = accuracy_score(gold, predicted)

    """precision_neg, recall_neg, f1_neg, _ = precision_recall_fscore_support(gold, predicted,
                                                                           average="binary", pos_label="negative")
    precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(gold, predicted,
                                                                           average="binary", pos_label="positive")"""
    if verbose:
        print("Precision (macro-average): ", precision_macro)
        print("Recall (macro-average): ", recall_macro)
        print("F1-Score (macro-average): ", f1_macro)
        print("Accuracy: ", acc_macro)

        """print()

        print("Precision (negative): ", precision_neg)
        print("Recall (negative): ", recall_neg)
        print("F1-Score (negative: ", f1_neg)

        print()

        print("Precision (positive): ", precision_pos)
        print("Recall (positive): ", recall_pos)
        print("F1-Score (positive): ", f1_pos)"""

    return precision_macro, recall_macro, f1_macro, acc_macro

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
    preprocessed_test = preprocess(test)

    print("English data:")

    C = 15.5

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
    C = 15.5
    pred2,_ = sentiment_classifier(train, labels, preprocessed_test2, C=C)
    print(len(pred2))
    print("\nEvaluation SVM")
    _, _, _, _ = evaluate(gold2, pred2, verbose=True)

    print("\nEvaluation Lexicon-based")
    lexicon_pred2 = lexicon_based_analyzer_eng(test2)
    _, _, _, _ = evaluate(gold2, lexicon_pred2, verbose=True)

    # test Google translations

    print("\n-----Google translations-----")

    test3, gold3 = [], []
    with open("../data/google_translated_german_test_data.csv", "r", encoding="UTF-8") as t2_file:
        csv_reader = csv.reader(t2_file, delimiter=";")

        for row in csv_reader:
            if len(row) == 2:
                test3.append(row[1])
                gold3.append(row[0])

    preprocessed_test3 = preprocess(test3)

    C = 15.5

    pred3,_ = sentiment_classifier(train, labels, preprocessed_test3, C=C)
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
    C = 15.5
    pred4, _ = sentiment_classifier(train, labels, preprocessed_test4, C=C)
    print("\nEvaluation SVM")
    _, _, _, _ = evaluate(gold4, pred4, verbose=True)

    print("\nEvaluation Lexicon-based")
    lexicon_pred4 = lexicon_based_analyzer_eng(test4)
    _, _, _, _ = evaluate(gold4, lexicon_pred4, verbose=True)


