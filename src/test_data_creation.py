"""
Author: Sarah Ackerschewski
Bachelor Thesis
WS 21/22

This file contains all methods for tokenizing,
randomizing and creating the test data for the sentiment analysis.
It also contains code for translating the German test data to English.
"""

from os import listdir
import random
from nltk import sent_tokenize
import csv
import deepl
import machine_translation as mt


def create_test():
    """

    :return:
    """
    # read negative reviews
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
                if token not in neg:
                    neg.append(token)

    label_neg = ["negative" for i in range(len(neg))]

    # read positive reviews
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
                if token not in pos:
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
            if len(sent) > 1:
                csv_writer.writerow([gold[i], sent])

    # read German reviews
    test = []
    with open("../data/test_movie_reviews_german.tsv", "r", encoding="UTF-8") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) == 3:
                tokenized = sent_tokenize(row[2], language="german")

                if row[1] in ["4", "5"]:
                    for sent in tokenized:
                        if len(sent) > 1:
                            test.append(("positive", sent))
                elif row[1] in ["1", "2"]:
                    for sent in tokenized:
                        if len(sent) > 1:
                            test.append(("negative", sent))
    random.shuffle(test)

    # write file with tokenized German test data
    with open('../data/test_movie_reviews_german_tokenized.csv', 'w', encoding="UTF-8") as out_file:
        csv_writer = csv.writer(out_file, delimiter=';')
        for l, sent in test:
            csv_writer.writerow([l, sent])


def translate(sentences, labels):
    """

    :param filename:
    :return:
    """
    with open("../data/translated_german_test_data.csv", "a", encoding="UTF-8") as file:
        csv_writer = csv.writer(file, delimiter=";")
        for i, text in enumerate(sentences):
            translator = deepl.Translator(auth_key="b1ba14d4-7232-ff86-ed94-3599ccfa25dd:fx")

            result = translator.translate_text(text, source_lang="DE", target_lang="EN-US")

            csv_writer.writerow([labels[i], result.text])


def translate_own_system(sentences, labels, mode):
    """

    :param sentences:
    :param labels:
    :return:
    """
    with open("../data/google_translated_german_test_data.csv", mode, encoding="UTF-8") as file:
        csv_writer = csv.writer(file, delimiter=";")
        translator, tokenizer = mt.translator()
        for i, text in enumerate(sentences):
            result = mt.predict_sequence(translator, tokenizer, text)
            print(result)
            csv_writer.writerow([labels[i], result])


def create_test_data_hotel():
    hotel_test = []
    hotel_gold = []
    with open("../data/test_hotel_reviews_english.txt") as h_file:
        for line in h_file.readlines():
            label = line[line.index("\"overall\": ") + 11:line.index("\"overall\": ") + 12]
            text = line[line.index("\"text\": ") + 8:line.index(", \"author\":")]
            tokenized = sent_tokenize(text, language="english")

            if label in ["4", "5"]:
                for sent in tokenized:
                    if len(sent) > 1:
                        hotel_test.append(sent)
                        hotel_gold.append("positive")
            elif label in ["1", "2"]:
                for sent in tokenized:
                    if len(sent) > 1:
                        hotel_test.append(sent)
                        hotel_gold.append("negative")

    randomized = list(zip(hotel_test, hotel_gold))
    random.shuffle(randomized)
    hotel_test, hotel_gold = zip(*randomized)

    # write file with tokenized German test data
    with open('../data/test_hotel_reviews_tokenized.csv', 'w', encoding="UTF-8") as out_file:
        csv_writer = csv.writer(out_file, delimiter=';')
        for i, sent in enumerate(hotel_test):
            csv_writer.writerow([hotel_gold[i], sent])


if __name__ == '__main__':
    # create_test()

    # create_test_data_hotel()
    # read German test data
    test2, gold2 = [], []
    neg, pos = 0,0
    with open("../data/test_movie_reviews_german_tokenized.csv", "r", encoding="UTF-8") as f:
        csv_reader = csv.reader(f, delimiter=";")

        for i, row in enumerate(csv_reader):
            if len(row) == 2:
                if row[0] == "negative":
                    if neg <= 1000:
                        gold2.append(row[0])
                        test2.append(row[1])

                elif row[0] == "positive":
                    if pos <= 1000:
                        gold2.append(row[0])
                        test2.append(row[1])
    # translate test data with Google Ajax Translation API
    length = 0
    translate_tuples = []
    trans_sent, trans_label = [], []
    for i in range(0, len(test2)):
        sent = test2[i]
        trans_sent.append(sent)
        trans_label.append(gold2[i])
        length += 1
        if length == 100:
            translate_tuples.append((trans_sent, trans_label))
            length = 0

    for i, tpl in enumerate(translate_tuples):
        if i == 0:
            translate_own_system(sentences=tpl[0], labels=tpl[1], mode="w+")
        else:
            translate_own_system(sentences=tpl[0], labels=tpl[1], mode="a")

    """
   # translate test data  with DeepL
 
    char_len = 0
    trans_sent, trans_label = [], []
    for i in range(0, len(test2)-2000):
        sent = test2[i]
        if char_len + len(sent) <= 500000:
            trans_sent.append(sent)
            trans_label.append(gold2[i])
            char_len += len(sent)
        else:
            break
 
    translation = translate(sentences=trans_sent, labels=trans_label)
    print(translation)"""
