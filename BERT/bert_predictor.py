import torch
import pandas as pd
import math
import click

from pytorch_transformers import BertConfig, BertTokenizer, BertForNextSentencePrediction
from tqdm import tqdm

from feature_extractor import *


class SampleType:
    text_a = ''
    text_b = None
    unique_id = 0


def get_batch(df, response_type):
    samples = []
    for i, row in df.iterrows():
        temp_sample = SampleType()
        temp_sample.unique_id = i
        temp_sample.text_a = row.context
        temp_sample.text_b = row[response_type]
        samples.append(temp_sample)
    return samples


@click.command()
@click.option('--data', default='validation_db.csv', help='Type of data to predict (val/test).')
@click.option('--dialogue_type', default='DB', type=click.Choice(['DB', 'normal']), help='Predict DB or not dialogues.')
@click.option('--dest', default='validation_db.txt', help='Prediction filename.')
@click.option('--batchsize', default=1, help='Batch prediction size.')
@click.option('--bert_model', default='bert-base-uncased', help='Batch prediction size.')
@click.option('--cuda', default=False, help='Use CUDA or nah fam.')
def start_inference(data, dialogue_type, dest, batchsize, bert_model, cuda):

    assert torch.cuda.is_available()!=True, 'PyTorch not running on GPU! #sadpanda'

    dialogue_type_dict = {'DB': 'db_response_new', 'normal': 'response'}

    config = BertConfig.from_pretrained(bert_model)
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model = BertForNextSentencePrediction(config)
    model.cuda()
    model.eval()

    df = pd.read_csv(data, usecols=['id'])
    df.dropna(inplace=True)
    row_count = df.shape[0]
    del df

    chunk_count = math.ceil(row_count/batchsize)

    with open(dest, 'w+'):
        pass

    cols = ['context', dialogue_type_dict[dialogue_type]]
    for chunk in tqdm(pd.read_csv(open(data, 'r'), usecols=cols, chunksize=batchsize),
                         desc='Batches', total=chunk_count):

        samples = get_batch(chunk, dialogue_type_dict[dialogue_type])

        assert len(samples)==chunk.shape[0], 'Some samples went missing!'

        if batchsize==1:
            results = convert_single_example_to_features(samples, tokenizer)
        else:
            results = convert_examples_to_features(samples, tokenizer)

        input_ids = torch.tensor([x.input_ids for x in results]).cuda()
        token_type_ids = torch.tensor([x.input_type_ids for x in results]).cuda()
        attention_mask = torch.tensor([x.input_mask for x in results]).cuda()

        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        outputs = torch.softmax(outputs, dim=1)
        db_probs = outputs[:, 1]

        with open(dest, 'a') as f:
            f.write('\n'.join([str(x) for x in db_probs.tolist()])+'\n')


if __name__=='__main__':
    start_inference()
