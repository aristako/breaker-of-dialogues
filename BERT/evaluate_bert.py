import torch
from pytorch_transformers import BertConfig, BertTokenizer, BertForNextSentencePrediction
import numpy as np
import pandas as pd

config = BertConfig.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction(config)
model.eval()
model.cuda()

df = pd.read_csv('breaker-of-dialogues/validation_db.csv')
max_word_count = 550


class SampleType:
    text_a = ''
    text_b = None
    unique_id = 0


def get_batch(df):
    samples = []
    for _, row in df.iterrows():
        temp_sample = SampleType()
        temp_sample.unique_id = row.id

        temp_sample.text_a = 'hello my name is lionel messi'
        temp_sample.text_b = 'and I play football'

        samples.append(temp_sample)



