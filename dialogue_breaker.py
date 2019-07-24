import pandas as pd
from tqdm import tqdm
import random


files_dict = {x: f'../data/reddit/{x}.csv' for x in ['validation', 'testing']}
subreddits = ['AskReddit', 'todayilearned', 'gaming', 'worldnews', 'food', 'science']

for data_type, filename in tqdm(files_dict.items(), desc='Data Type Creation'):
    tqdm.write(f'Breaking {data_type} dialogues.')

    df = pd.read_csv(filename)

    df_length = df.shape[0]
    random.seed(10)
    random_seeds = random.sample(range(df_length), df_length)
    df['random_seed'] = random_seeds

    df.dropna(subset=['context', 'response'], inplace=True)
    df = df[df.subreddit.isin(subreddits)]
    df.sort_values(by=['subreddit'], inplace=True)
    subreddit_breakdown = df.subreddit.value_counts().sort_index().to_dict()
    # df['pool'] = df['context'] + ' #_new_utterance_# ' + df['response']
    db_subreddits, db_responses, db_responses_new = [], [], []
    for k, v in tqdm(subreddit_breakdown.items(), desc='Batch Breaking Dialogues'):
        try:
            pool = df[df.subreddit!=k].sample(v, random_state=999)[['subreddit', 'context', 'response', 'random_seed']]
        except ValueError:
            pool = df[df.subreddit!=k].sample(v, random_state=999, replace=True)[['subreddit', 'context', 'response',
                                                                                  'random_seed']]
        db_subreddits.extend(pool.subreddit.tolist())
        db_responses.extend(pool.response.tolist())

        for _, row in pool.iterrows():
            random.seed(row.random_seed)
            db_responses_new.append(random.sample(row.context.split('#_new_utterance_#'), 1)[0])

    df['db_subreddit'] = db_subreddits
    df['db_response'] = db_responses
    df['db_response_new'] = db_responses_new

    df.to_csv(f'{data_type}_db.csv')






    # db_subreddit, db_response = [], []
    # for k, v in tqdm(subreddit_breakdown.items(), desc='Batch Breaking Dialogues'):
    #     try:
    #         pool = df[df.subreddit!=k].sample(v, random_state=999)[['subreddit', 'response']]
    #     except ValueError:
    #         pool = df[df.subreddit!=k].sample(v, random_state=999, replace=True)[['subreddit', 'response']]
    #     db_subreddit.extend(pool.subreddit.tolist())
    #     db_response.extend(pool.response.tolist())
    #
    # df['db_subreddit'] = db_subreddit
    # df['db_response'] = db_response
    #
    # df.to_csv(f'{data_type}_db.csv')

