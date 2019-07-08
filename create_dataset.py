from google.cloud import storage
from helpers import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
random.seed(8)


def download_blob(src, dest):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('raw-reddit-dataset-bucket')
    blob = bucket.blob(src)

    blob.download_to_filename(dest)


def create_split(filenames, destination, subreddit_name, create_file=True):
    if create_file:
        df = pd.DataFrame(columns=['id', 'subreddit', 'context', 'response', 'utterance_count',
                                   'word_count', 'char_count'])
        df.to_csv(f'{destination}.csv', index=False)

    path = 'raw-reddit-dataset-bucket/reddit/20190614'

    for filename in tqdm(filenames, desc='Files', position=1, leave=True):

        ids, contexts, responses, subreddits = [], [], [], []
        utterance_counts, word_counts, char_counts = [], [], []

        download_blob(src=f'{path}/{filename}', dest='tmp/temp.tfrecords')
        samples = obtain_samples('tmp/temp.tfrecords')

        for sample in tqdm(samples, desc='Dialogues', position=2, leave=False):
            dialogue = []
            thread_id = get_feature_value(sample, 'thread_id')
            subreddit = get_feature_value(sample, 'subreddit')
            dialogue.append(clean_text(get_feature_value(sample, 'response')))
            dialogue.append(get_feature_value(sample, 'context'))
            dialogue.extend(get_additional_context(sample))
            dialogue = dialogue[::-1]

            if len(dialogue) > 3 and subreddit.lower() == subreddit_name.lower():
                ids.append(thread_id)
                subreddits.append(subreddit)
                contexts.append(' #_new_utterance_# '.join(dialogue[:-1]))
                responses.append(dialogue[-1])
                utterance_counts.append(len(dialogue))
                word_counts.append(np.sum([len(x.split()) for x in dialogue]))
                char_counts.append(np.sum([len(x) for x in dialogue]))

        temp = pd.DataFrame()
        temp['id'] = ids
        temp['subreddit'] = subreddits
        temp['context'] = contexts
        temp['response'] = responses
        temp['utterance_count'] = utterance_counts
        temp['word_count'] = word_counts
        temp['char_counts'] = char_counts
        temp.set_index('id')
        temp.to_csv(f'{destination}.csv', mode='a', header=False, index=False)



