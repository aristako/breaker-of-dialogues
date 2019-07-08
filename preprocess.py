import pandas as pd
from tqdm import tqdm
import math
# need to preprocess data both for StarSpace and for BERT architectures


def batch_process(filename, destination):
    df = pd.read_csv(open(filename, 'r'), usecols=['id'])
    num_of_lines = df.shape[0]
    chunksize = 10 ** 6
    batch_count = math.ceil(num_of_lines / chunksize)

    cols = ['context', 'response']
    for chunk in tqdm(pd.read_csv(open(filename, 'r'), usecols=cols, chunksize=chunksize),
                      desc='Batches', total=batch_count):
        chunk['joined'] = chunk['context'] + '\t' + chunk['response']
        samples = '\n'.join(list(chunk['joined'].astype(str)))
        with open(destination, 'a') as f:
            f.write(samples)


filenames = [f'train_0{i}_of_04.csv' for i in range(1,5)]
for batch in tqdm(filenames, desc='Files'):
    batch_process(batch, 'starspace_train.txt')
