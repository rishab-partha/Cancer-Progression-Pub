import numpy as np
from pathlib import Path
import collections
import itertools
import os
import drugs

datasetpath = Path("Dataset")
datasetpath = datasetpath / "TCGADataset"
allclasses = [f.name for f in os.scandir(datasetpath) if f.is_dir()]

def load_heatmap(classname):
    classpath = datasetpath / classname
    mutationspath = classpath / "data_mutations_extended.txt"
    clinicalpath = classpath / "data_clinical_patient.txt"
    mutations = np.loadtxt(mutationspath, dtype=str, skiprows=1, delimiter='\t', usecols=(16,0))
    clinical = np.loadtxt(clinicalpath, dtype=object, skiprows=5, delimiter='\t', usecols= (0, 6))
    
    cntr = 0
    while cntr < clinical.shape[0]:
        clinical[cntr][1] = clinical[cntr][1][6:]
        if not clinical[cntr][1] == '':
            if not (clinical[cntr][1][-1] == 'I' or clinical[cntr][1][-1] == 'V'):
                clinical[cntr][1] = clinical[cntr][1][:-1]
        cntr += 1

    print("Stage Names Processed")
    
    mutationlist = sorted(set(list(mutations[:, 1])))
    mutationdtype = [('Gene Name', object), ('Stage I Total', np.int32), ('Stage I Patient', np.int32), ('Stage II Total', np.int32), ('Stage II Patient', np.int32), ('Stage III Total', np.int32),
    ('Stage III Patient', np.int32), ('Stage IV Total', np.int32), ('Stage IV Patient', np.int32), ('Overall Total', np.int32), ('Overall Patient', np.int32)]
    mutationoccurences = np.zeros((len(mutationlist) + 1,), dtype=mutationdtype)

    mutationdict = {}
    for i in range(len(mutationlist)):
        mutationdict[mutationlist[i]] = i + 1
        mutationoccurences[i + 1][0] = mutationlist[i]
    mutationoccurences[0][0] = "Total Patient #"
    print(mutationoccurences[0])

    print("Mutation IDs Processed")

    cntr = 0
    curpatient = ''
    seenmutationset = set()
    for mutation in mutations:
        if not mutation[0][:-3] == curpatient:
            seenmutationset.clear()
            curpatient = mutation[0][:-3]
            while not clinical[cntr][0] == curpatient:
                cntr += 1
            if not clinical[cntr][1] == '':
                mutationoccurences[0][9] += 1
                mutationoccurences[0][10] += 1
                if clinical[cntr][1] == 'I':
                    mutationoccurences[0][1] += 1
                    mutationoccurences[0][2] += 1
                elif clinical[cntr][1] == 'II':
                    mutationoccurences[0][3] += 1
                    mutationoccurences[0][4] += 1
                elif clinical[cntr][1] == 'III':
                    mutationoccurences[0][5] += 1
                    mutationoccurences[0][6] += 1
                else:
                    mutationoccurences[0][7] += 1
                    mutationoccurences[0][8] += 1
        
        index = mutationdict[mutation[1]]
        seenmutation = mutation[1] in seenmutationset

        if not seenmutation:
            seenmutationset.add(mutation[1])

        if not clinical[cntr][1] == '':
            mutationoccurences[index][9] += 1
            if not seenmutation:
                mutationoccurences[index][10] += 1
            if clinical[cntr][1] == 'I':
                mutationoccurences[index][1] += 1
                if not seenmutation:
                    mutationoccurences[index][2] += 1
            elif clinical[cntr][1] == 'II':
                mutationoccurences[index][3] += 1
                if not seenmutation:
                    mutationoccurences[index][4] += 1
            elif clinical[cntr][1] == 'III':
                mutationoccurences[index][5] += 1
                if not seenmutation:
                    mutationoccurences[index][6] += 1
            else:
                mutationoccurences[index][7] += 1
                if not seenmutation:
                    mutationoccurences[index][8] += 1

    print("Mutation Occurences Processed")
    return mutationoccurences

def save_heatmap(classname):
    heatmap = load_heatmap(classname)
    name = classname + 'heatmap.csv'
    np.savetxt(name, heatmap, delimiter = ',', fmt = ['%s', '%d', '%d', '%d', '%d', '%d', '%d', '%d', '%d', '%d', '%d'], 
    header = 'Gene Name,Stage I Total,Stage I Patient,Stage II Total,Stage II Patient,Stage III Total,Stage III Patient,Stage IV Total,Stage IV Patient,Overall Total,Overall Patient', comments='')

def filter_genes_from_heatmap(classname, ignore_stages, gene_num = 100, sort_indices = list([10, 2, 4, 6, 8]), col_names = ['Gene Name', 'Stage I Total', 'Stage I Patient', 'Stage II Total', 'Stage II Patient', 
'Stage III Total', 'Stage III Patient', 'Stage IV Total', 'Stage IV Patient', 'Overall Total', 'Overall Patient']):

    heatmap = load_heatmap(classname)
    mutationlist = list()
    for i in range(len(sort_indices)):
        if i in ignore_stages:
            continue

        heatmap = heatmap[np.argsort(heatmap[col_names[sort_indices[i]]])]
        heatmap = np.flipud(heatmap)
        for j in range(1, (gene_num + 1)):
            mutationlist.append(heatmap[j][0])

    mutationlist = list(dict.fromkeys(mutationlist))
    mutationset = set(mutationlist)
    return mutationlist, mutationset

def filter_genes_and_drugs_from_heatmap(classname, ignore_stages, gene_num = 100, sort_indices = list([2, 4, 6, 8]), col_names = ['Gene Name', 'Stage I Total', 'Stage I Patient', 'Stage II Total', 'Stage II Patient', 
'Stage III Total', 'Stage III Patient', 'Stage IV Total', 'Stage IV Patient', 'Overall Total', 'Overall Patient']):

    heatmap = load_heatmap(classname)
    mutation_list_and_drugs = np.empty((4 - len(ignore_stages), gene_num, 2), dtype = object)
    ignored_cntr = 0
    mutation_frequencies = np.empty((4 - len(ignore_stages), ), dtype = object)
    for i in range(len(sort_indices)):
        if (i + 1) in ignore_stages:
            ignored_cntr += 1
            continue

        mutation_frequencies[i - ignored_cntr] = dict()
        heatmap = heatmap[np.argsort(heatmap[col_names[sort_indices[i]]])]
        heatmap = np.flipud(heatmap)
        for j in range(1, (gene_num + 1)):
            mutation_list_and_drugs[i - ignored_cntr][j - 1][0] = heatmap[j][0]
            mutation_list_and_drugs[i - ignored_cntr][j - 1][1] = drugs.findpossibledrugs(heatmap[j][0])
            mutation_frequencies[i - ignored_cntr][heatmap[j][0]] = heatmap[j][sort_indices[i]] / heatmap[0][sort_indices[i]]
    
    return mutation_list_and_drugs, mutation_frequencies

def save_all_heatmaps():
    for s in allclasses:
        save_heatmap(s)

def open_one_class(classname):
    classpath = datasetpath / classname
    mutationspath = classpath / "data_mutations_extended.txt"
    clinicalpath = classpath / "data_clinical_patient.txt"
    mutations = np.loadtxt(mutationspath, dtype=str, skiprows=1, delimiter='\t', usecols=(16, 0))
    clinical = np.loadtxt(clinicalpath, dtype=object, skiprows=5, delimiter='\t', usecols = (0, 6, 4, 5, 25))
    hispanic = np.loadtxt(clinicalpath, dtype = object, skiprows = 5, delimiter = '\t', usecols = (11))

    for i in range(clinical.shape[0]):
        if hispanic[i] == "Hispanic Or Latino":
            clinical[i][4] = "Hispanic or Latino"
        clinical[i][1] = clinical[i][1][6:]
        if not clinical[i][1] == '':
            if not (clinical[i][1][-1] == 'I' or clinical[i][1][-1] == 'V'):
                clinical[i][1] = clinical[i][1][:-1]

            if not clinical[i][1] == '':
                if not (clinical[i][1][-1] == 'I' or clinical[i][1][-1] == 'V'):
                    clinical[i][1] = ''
                    
        clinical[i][2] = clinical[i][2][:2]
        if not clinical[i][4] == '':
            clinical[i][4] = clinical[i][4].split()[0]
    
    return mutations, clinical

def loadoneclass(classname):
    mutations, clinical = open_one_class(classname)
    alldata = np.insert(clinical, 1, '', axis = 1)
    curpatient = mutations[0][0][:-3]
    genestring = ''
    cntr = 0
    for mutation in mutations:
        if not mutation[0][:-3] == curpatient:
            while not curpatient == alldata[cntr][0]:
                alldata = np.delete(alldata, (cntr), axis = 0)
            if genestring == '':
                alldata = np.delete(alldata, (cntr), axis = 0)
            else:
                alldata[cntr][1] = genestring
                genestring = ''
                cntr += 1
            curpatient = mutation[0][:-3]

        if not genestring == '':
            genestring += ' '
        genestring += mutation[1]

    if not genestring == '':
        alldata[cntr][1] = genestring
        cntr += 1

    while cntr < alldata.shape[0]:
        alldata = np.delete(alldata,(cntr), axis = 0)

    mutationlist = list(mutations[:, 1])
    return alldata, mutationlist

def load_one_class_from_heatmap(classname, ignore_stages, gene_num = 100, classes = ['I', 'II', 'III', 'IV']):
    mutations, clinical = open_one_class(classname)
    
    for i in range(clinical.shape[0]):
        for j in ignore_stages:
            if clinical[i][1] == classes[j - 1]:
                clinical[i][1] = ''
    
    mutationlist, mutationset = filter_genes_from_heatmap(classname = classname, ignore_stages = ignore_stages, gene_num=gene_num)

    alldata = np.insert(clinical, 1, '', axis = 1)
    curpatient = mutations[0][0][:-3]
    genestring = ''
    cntr = 0
    lencntr = 0
    overalllencntr = 0
    for mutation in mutations:
        if not mutation[0][:-3] == curpatient:
            while not curpatient == alldata[cntr][0]:
                alldata = np.delete(alldata, (cntr), axis = 0)
            if genestring == '':
                alldata = np.delete(alldata, (cntr), axis = 0)
            else:
                alldata[cntr][1] = genestring
                genestring = ''
                overalllencntr = max(overalllencntr, lencntr)
                lencntr = 0
                cntr += 1
            curpatient = mutation[0][:-3]

        if mutation[1] in mutationset:
            if not genestring == '':
                genestring += ' '
            genestring += mutation[1]
            lencntr += 1
    
    if not genestring == '':
        alldata[cntr][1] = genestring
        overalllencntr = max(overalllencntr, lencntr)
        cntr += 1
    
    print(overalllencntr)
    while cntr < alldata.shape[0]:
        alldata = np.delete(alldata, (cntr), axis = 0)
    
    return alldata, mutationlist


def reducelist(mutationlist):
    counts = collections.Counter(mutationlist)
    freqsortmutlist = np.asarray(list(dict.fromkeys(sorted(mutationlist, key=counts.get, reverse=True))))
    return freqsortmutlist

def processoneclass(classname):
    data, mutationlist = loadoneclass(classname)
    mutationlist = reducelist(mutationlist)
    return data, mutationlist, len(mutationlist)

def processallmutations():
    alldata = np.empty([0, 7], dtype = object)
    mutationlist = []
    for s in allclasses:
        print("Loading " + s)
        oneclassdata, oneclassmutationlist = loadoneclass(s)
        oneclassdata = np.insert(oneclassdata, 6, s, axis = 1)
        alldata = np.append(alldata, oneclassdata, axis = 0)
        mutationlist = list(itertools.chain(mutationlist, oneclassmutationlist))
    
    mutationlist = reducelist(mutationlist)
    return alldata, mutationlist, len(mutationlist)

def processonecolumn(alldata, mutationlist, colnum):
    collist = []
    cntr = 0
    while cntr < alldata.shape[0]:
        if alldata[cntr][colnum] == '':
            alldata = np.delete(alldata, (cntr), axis = 0)
        else:
            alldata[cntr][1] = alldata[cntr][colnum] + ' ' + alldata[cntr][1]
            collist.append(alldata[cntr][colnum])
            cntr += 1
    
    collist = reducelist(collist)
    mutationlist = list(itertools.chain(mutationlist, collist))
    return alldata, mutationlist

def processforstageoutput(alldata):
    stagelist = sorted(set(list(alldata[:, 2])))
    if stagelist[0] == '':
        stagelist = stagelist[1:]
    cntr = 0
    while cntr < alldata.shape[0]:
        if alldata[cntr][2] == '':
            alldata = np.delete(alldata, (cntr), axis = 0)
        else:
            for i in range(len(stagelist)):
                if alldata[cntr][2] == stagelist[i]:
                    alldata[cntr][2] = i
                    break
            cntr += 1

    return alldata, stagelist
                

def processstagenum(alldata, mutationlist):
    return processonecolumn(alldata, mutationlist, 2)

def processage(alldata, mutationlist):
    return processonecolumn(alldata, mutationlist, 3)

def processgender(alldata, mutationlist):
    return processonecolumn(alldata, mutationlist, 4)

def processrace(alldata, mutationlist):
    return processonecolumn(alldata, mutationlist, 5)

def processcancertype(alldata, mutationlist):
    return processonecolumn(alldata, mutationlist, 6)

    
def test_tcga_loader():
    classname = 'LUAD'
    print("Loading " + classname)
    sampledata, samplemutlist, mutlength = processoneclass(classname)
    processeddata, processedmutlist = processage(sampledata, samplemutlist)
    print(sampledata[0])
    print(processeddata[0])
    print(samplemutlist[:20])
    print(processedmutlist[:20])
    print(mutlength)

    print("Loading Whole Dataset")
    alldata, allmutlist, mutlength = processallmutations()
    print(alldata[0])
    print(alldata[8])
    print(allmutlist[:20])

""" mutationlist, overallmutlist = filter_genes_and_drugs_from_heatmap('HNSC', {1}, gene_num = 200)
print(mutationlist[1][:20])
print(overallmutlist[:5])
alldata, mutationlist = load_one_class_from_heatmap(classname = 'THCA', ignore_stages = {}, gene_num = 200)
print(mutationlist)
print(len(mutationlist)) """