# Bilingual Sentiment Analysis

This project contains code for using German movie reviews translated to English data as test data for a sentiment analyzer to compare if translation loss has an effect on polarity detection

## Data
The [training data](https://github.com/sarahackerschewski/sentiment-translation-loss/blob/main/data/training_movie_reviews.txt) is taken from the [VADER project](https://github.com/cjhutto/vaderSentiment/tree/master/additional_resources) by Hutto and Gilbert (2018). It is structured into item no., intensity rating and English sentence.

The English_test_data is taken from the [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/), where each review is a single file, having the gold polarity in the filename.

The original [German test data](https://github.com/sarahackerschewski/sentiment-translation-loss/blob/main/data/test_movie_reviews_german.tsv) can be found [here](https://zenodo.org/record/3693810/#.Yg1DpN_MLDd) in the sentiment-data-reviews-and-neutral.zip, which is tab-separated and contains the link to the review, the star rating from 1(bad) to 10(good) and the whole review in each row.

The training set contains 10605 sentences. The comprised and [tokenized English test data](https://github.com/sarahackerschewski/sentiment-translation-loss/blob/main/data/test_movie_reviews_english_tokenized.csv) contains 5000 positve and 5000 negative sentences. The comprised and [tokenized German test data](https://github.com/sarahackerschewski/sentiment-translation-loss/blob/main/data/test_movie_reviews_german_tokenized.csv) contains 36.481 sentences.

The [translated German test data](https://github.com/sarahackerschewski/sentiment-translation-loss/blob/main/data/translated_german_test_data.csv) contains 1000 positive and 1000 negative sentences translated from German to English with the [DeepL API](https://www.deepl.com/pro-api?cta=menu-pro-api). Similarly, the [other translation test data](https://github.com/sarahackerschewski/sentiment-translation-loss/blob/main/data/other_translated_german_test_data.csv) contains 7367 review sentence snippets which were translated as explained in [this article](https://towardsdatascience.com/machine-translation-with-transformers-using-pytorch-f121fe0ad97b) with a [pre-trained MT model](https://huggingface.co/Helsinki-NLP/opus-mt-de-en) found on huggingface.

## SVM
The [sentiment-analyzer.py](https://github.com/sarahackerschewski/sentiment-translation-loss/blob/main/src/sentiment-analysis.py) contains the preprocessing of the data (Lemmatizing, removal of HTML tags, TF-IDF, chi-squared feature selection) and a sentiment analyzer, which is built with a LinearSVM. The sentiment analyzer can be used by preprocessing the test data and then adding it as a parameter for the sentiment_classifier function. It returns the predictions but also the 10-fold CV results as a tuple.
```python
xtest = preprocess(xtest)
predictions, cv_tpl = sentiment_classifier(xtrain, ytrain, xtest)
```
It is also possible to evaluate the predictions of the sentiment analyzer if gold labels exist. 
```python
 precision, recall, f1, _ = evaluate(ytest, predictions, verbose=True)
```

The test data is properly created (tokenized etc.) in the [test_creation.py](https://github.com/sarahackerschewski/sentiment-translation-loss/blob/main/src/test_data_creation.py), which also comprises the code for translating from German to English (DeepL & transformer MT system).

The transformer MT system can be found in [machine_translation.py](https://github.com/sarahackerschewski/sentiment-translation-loss/blob/main/src/machine_translation.py)

## Results
It is possible to save the prediction results of the sentiment analyzer by setting the parameter of the sentiment analyzer function to True and adding the gold_labels and the filename. 
```python
predictions, cv_tpl = sentiment_classifier(xtrain, ytrain, xtest, save_pred=True, y_test=ytest, filename="de-en")
```
The resulting files can be found in the [results directory](https://github.com/sarahackerschewski/sentiment-translation-loss/tree/main/results)
