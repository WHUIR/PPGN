'''extracting overlapping users'''
import pandas as pd
import numpy as np
import gzip, os


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    df = {}
    for i,d in enumerate(parse(path)):
        df[i] = d

    return pd.DataFrame.from_dict(df, orient='index')


def construct(path_s, path_t):
    s_5core, t_5core = getDF(path_s), getDF(path_t)
    s_users = set(s_5core['reviewerID'].tolist())
    t_users = set(t_5core['reviewerID'].tolist())
    overlapping_users = s_users & t_users

    s = s_5core[s_5core['reviewerID'].isin(overlapping_users)][['reviewerID','asin','overall','unixReviewTime']]
    t = t_5core[t_5core['reviewerID'].isin(overlapping_users)][['reviewerID','asin','overall','unixReviewTime']]

    csv_path_s = path_s.replace('reviews_','').replace('_5.json.gz','.csv')
    csv_path_t = path_t.replace('reviews_','').replace('_5.json.gz','.csv')
    s.to_csv(csv_path_s, index=False)
    t.to_csv(csv_path_t, index=False)

    print('Build raw data to %s.' % csv_path_s)
    print('Build raw data to %s.' % csv_path_t)


if __name__ == '__main__':
    construct('../data/cds_music/reviews_CDs_and_Vinyl_5.json.gz', '../data/cds_music/reviews_Digital_Music_5.json.gz')
