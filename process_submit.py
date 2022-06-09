from lib2to3.pgen2.pgen import DFAState
import os
from tqdm import tqdm
import pandas as pd
import argparse
import pdb
from rank_bm25 import BM25Okapi
from itertools import combinations
from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--code_path' ,  type = str)
parser.add_argument('--save_path' ,  type = str)


args = parser.parse_args()



def preprocess_script(file):
    lines = file.split("\n")
    preproc_lines = []
    for line in lines:
        if line.lstrip().startswith('#'):
            continue
        line = line.rstrip()
        if '#' in line:
            line = line[:line.index('#')]
        line = line.replace('\n','')
        line = line.replace('    ','\t')
        if line == '':
            continue
        preproc_lines.append(line)
    preprocessed_script = '\n'.join(preproc_lines)
    return preprocessed_script

if __name__ == '__main__':
    df = pd.read_csv(args.code_path)
    df['code1'] = df['code1'].apply(preprocess_script)
    df['code2'] = df['code2'].apply(preprocess_script)
    df['similar'] = -1

    df = df.reset_index(drop=True)
    df.to_csv(args.save_path +"/test.csv")
    




