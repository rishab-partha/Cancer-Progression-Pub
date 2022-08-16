import pandas as pd 
import numpy as np
from pathlib import Path

datasetpath = Path("Dataset")

drugbankpath = datasetpath / "DrugBankFiles"
drugbanktargetpath = drugbankpath / "Target IDs" / "all.csv"
drugbankidpath = drugbankpath / "Drug IDs" / "drugbank vocabulary.csv"

bpsiupharpath = datasetpath / "BPS-IUPHAR-Files" / "interactions.csv"

drugbanktargets = pd.read_csv(drugbanktargetpath, usecols = [2, 12])
drugbankids = pd.read_csv(drugbankidpath, index_col = 0, usecols = [0, 2])
bpsiuphar = pd.read_csv(bpsiupharpath, index_col = 1, usecols = [2, 12])
bpsiuphar.dropna()

def findpossibledrugs(gene):
    druglist = []
    drugbankgenetargets = drugbanktargets[drugbanktargets['Gene Name'] == gene]

    for i in range(drugbankgenetargets.shape[0]):
        for drugid in drugbankgenetargets.iat[i, 1].split('; '):
            drugname = drugbankids.at[drugid, 'Common name']
            if drugname.lower() in bpsiuphar.index:
                bpsiuphardrugtargets = bpsiuphar.at[drugname.lower(), 'target_gene_symbol']

                if isinstance(bpsiuphardrugtargets, float):
                    continue

                if isinstance(bpsiuphardrugtargets, str):
                    if bpsiuphardrugtargets == gene:
                        druglist.append(drugname)
                    continue

                for j in range(bpsiuphardrugtargets.shape[0]):
                    geneid = bpsiuphardrugtargets.iat[j]
                    if gene == geneid:
                        druglist.append(drugname)
                        continue
            else:
                druglist.append(drugname)

    druglist = list(set(druglist))
    druglist.sort()
    return druglist

print(findpossibledrugs("DMD"))
    