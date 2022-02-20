"""
Author: Sarah Ackerschewski
Bachelor Thesis
WS 21/22
Supervisor: Cagri Cöltekin

This file contains the code for a machine translation system
based on the article found on
https://towardsdatascience.com/machine-translation-with-transformers-using-pytorch-f121fe0ad97b (last accessed: 20.02.22)
and with the pre-trained model from huggingface
(see https://huggingface.co/Helsinki-NLP/opus-mt-de-en) (last accessed: 20.02.22)
"""

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate(text):
    """
    uses a Pytorch Transformer to make use of the HuggingFace pre-trained German-to-English machine translation model
    :param text: sequence to translate
    :return: translated sequence
    """
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    translation = pipeline("translation_de_to_en", model=model, tokenizer=tokenizer)

    return translation(text)[0]['translation_text']


