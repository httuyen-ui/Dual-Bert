"""
Tim index cua mot thuoc trong drug_SMILES_750.csv.
Dung de xac dinh dung drug_index cho case study.

Vi du:
    python find_drug.py --query ibuprofen
    python find_drug.py --query "CC(C)Cc1ccc"     # tim theo SMILES
    python find_drug.py --show 199                 # xem thuoc o index 199
"""
import argparse
import pandas as pd
from smiles2vector import load_drug_smile

parser = argparse.ArgumentParser()
parser.add_argument('--smiles_file', default='data/drug_SMILES_750.csv')
parser.add_argument('--query', default='', help='Ten thuoc hoac chuoi SMILES con')
parser.add_argument('--show', type=int, default=-1, help='In thuoc o index nay')
args = parser.parse_args()

# 1. Xem cau truc CSV
df = pd.read_csv(args.smiles_file)
print('Cot:', df.columns.tolist())
print('Shape:', df.shape)
print(df.head(3).to_string())
print('-' * 60)

# 2. drug_smile list (index = drug_id dung trong drug_side)
drug_dict, drug_smile = load_drug_smile(args.smiles_file)

if args.show >= 0:
    print('index {} -> SMILES: {}'.format(args.show, drug_smile[args.show]))

if args.query:
    q = args.query.lower()
    found = False
    # tim trong moi cot cua df (ten thuoc neu co)
    for i in range(len(df)):
        row_text = ' '.join(str(x) for x in df.iloc[i].tolist()).lower()
        if q in row_text:
            print('[df row {}] {}'.format(i, df.iloc[i].tolist()))
            found = True
    # tim trong SMILES list
    for i, sm in enumerate(drug_smile):
        if q in str(sm).lower():
            print('[drug_smile index {}] {}'.format(i, sm))
            found = True
    if not found:
        print('Khong tim thay "{}". Thu chuoi ngan hon hoac kiem tra cot ten.'.format(args.query))
