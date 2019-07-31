import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForNextSentencePrediction
import click
from tqdm import tqdm
import pickle


def bert_prediction(context, response, model, tokenizer):
    tokenized_context = tokenizer.tokenize('[CLS] ' + context + ' [SEP]')
    tokenized_response = tokenizer.tokenize(response + ' [SEP]')

    tokenized_text = tokenized_context + tokenized_response
    # if too long, drop leading tokens
    if len(tokenized_text) > 512:
        tokenized_text = tokenized_text[-512:]

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [0] * len(tokenized_context) + [1] * len(tokenized_response)
    with torch.no_grad():
        tokens_tensor = torch.tensor([indexed_tokens]).cuda()
        segments_tensors = torch.tensor([segments_ids]).cuda()
        res = torch.softmax(model(tokens_tensor, segments_tensors), dim=1).detach().cpu().numpy()[:, 1][0]
    return res


@click.command()
@click.option('--bert_model', default='bert-base-uncased', help='Batch prediction size.')
@click.option('--batch_count', default=-1, type=int, help='Which batch to process for hacky "multiprocessing".')
def start_inference(batch_count, bert_model):
    torch.manual_seed(10)

    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model = BertForNextSentencePrediction.from_pretrained(bert_model)
    model.eval()
    model.cuda()

    if batch_count != -1:
        batch_size = 125000
        df = pd.read_csv('validation_db.csv', skiprows=range(1, batch_count * batch_size + 1), nrows=batch_size)
        print(f'About to process batch number {batch_count}, which contains {df.shape[0]} samples.')
    else:
        df = pd.read_csv('validation_db.csv')

    normal_probs_single, db_probs_single = [], []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        context = row.context.split('#_new_utterance_#')[-1]
        normal_probs_single.append(bert_prediction(context, row.response, model, tokenizer))
        db_probs_single.append(bert_prediction(context, row.db_response_new, model, tokenizer))

    with open(f'part_{batch_count:02}_normal.pkl', 'wb') as f:
        pickle.dump(normal_probs_single, f)

    with open(f'part_{batch_count:02}_db.pkl', 'wb') as f:
        pickle.dump(db_probs_single, f)


if __name__ == '__main__':
    start_inference()
