import tensorflow as tf 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, enable=True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import matplotlib.pyplot as plt
import drugs
import load_tcga
import numpy as np
import os
import time
from pathlib import Path
import pandas as pd
import collections
import itertools
import seaborn as sns
import load_tcga as lt 
import networks as nn
import drugs

models_path = Path("models")
data_len = {'BLCA': 514, 'BRCA' : 853, 'COADREAD' : 1962, 'HNSC' : 422, 'KIRC' : 133, 'LIHC' : 277, 'LUAD' : 439, 'LUSC' : 302, 'SKCM' : 8665, 'STAD' : 1649, 'THCA' : 155}
ignored_stages = {'BLCA': {1}, 'BRCA': {4}, 'COADREAD': {}, 'HNSC': {1}, 'KIRC':{}, 'LIHC': {4}, 'LUAD' : {4}, 'LUSC' : {4}, 'SKCM' : {}, 
                 'STAD' : {}, 'THCA' : {}}

stage_lists = {'BRCA' : ['I', 'II', 'III'], 'COADREAD' : ['I', 'II', 'III', 'IV'], 'HNSC' : ['II', 'III', 'IV'], 'KIRC' : ['I', 'II', 'III', 'IV'], 'LIHC' : ['I', 'II', 'III'], 'LUAD' : ['I', 'II', 'III'],' LUSC' : ['I', 'II', 'III'],
               'SKCM' : ['I', 'II', 'III', 'IV'], 'STAD' : ['I', 'II', 'III', 'IV'], 'THCA' : ['I', 'II', 'III', 'IV']}

def load_model(cancer_type):
    model_id = cancer_type + "_stage_model_0.8_Top200Preprocessed_50Epochs"
    model_path = models_path / cancer_type / model_id
    model = tf.keras.models.load_model(model_path)
    return model

cancer_type = 'BRCA'
print("Loading Models...")
model = load_model(cancer_type)
print("Loading Heatmaps...")
mutationlist_and_drugs, mutation_frequencies = lt.filter_genes_and_drugs_from_heatmap(cancer_type, ignored_stages[cancer_type], gene_num=200)

def process_one_sample(gene_seq, stage_id, heatmap_drug_data):
    # print("One Iteration")
    # print(gene_seq)
    gene_list = gene_seq.astype('str').tolist()
    # print(gene_list)
    gene_set = set(gene_list)
    # print(gene_set)

    if '' in gene_set:
        gene_set.remove('')

    # print(gene_set)

    predicted_mutations = list()
    mutation_drugs = list()

    for i in range(heatmap_drug_data.shape[1]):
        gene_name = heatmap_drug_data[stage_id][i][0]

        if gene_name in gene_set:
            # print('hello')
            continue

        predicted_mutations.append(gene_name)
        mutation_drugs.append(heatmap_drug_data[stage_id][i][1])

    return predicted_mutations, mutation_drugs

def process_sample_set(gene_seqs, stage_ids, heatmap_drug_data):
    if not gene_seqs.shape[0] == stage_ids.shape[0]:
        raise RuntimeError("Mismatched array size")
    
    predicted_mutations = np.empty((gene_seqs.shape[0]), dtype = object)
    mutation_drugs = np.empty((gene_seqs.shape[0]), dtype = object)

    for i in range(gene_seqs.shape[0]):
        predicted_mutations[i], mutation_drugs[i] = process_one_sample(gene_seqs[i], stage_ids[i], heatmap_drug_data)

    return predicted_mutations, mutation_drugs

def predict_specific_instance():
    print('What Cancer Type?')
    cancer_type = input()

    print()
    print('Loading Model...')

    print()
    print('Loading Mutations/Drugs')

    print()
    print("Please input your mutation sequence (space separated):")
    gene_seq = input()

    gene_seq = tf.strings.split(gene_seq).numpy()
    gene_seq = np.reshape(gene_seq, (1, gene_seq.shape[0]))

    Y_pred = model(gene_seq)
    Y_pred = tf.nn.softmax(Y_pred)
    stage_id = np.argmax(Y_pred, axis = 1)[0]
    stage_name = stage_lists[cancer_type][stage_id]

    print()
    print("Your Cancer is Stage ", stage_name)
    gene_seq = np.reshape(gene_seq, (gene_seq.shape[1], ))
    predicted_mutations, drug_lists = process_one_sample(gene_seq, stage_id, mutationlist_and_drugs)
    top_ten_mutations = predicted_mutations[:10]
    merged_drug_list = list()
    for i in range(10):
        merged_drug_list.extend(drug_lists[i])

    print()
    print("Your Ten Most Likely Future Mutations Are:")
    print(', '.join(top_ten_mutations))
    print()
    print("Recommended Drug Treatments:")
    print(', '.join(merged_drug_list))