# Details

Public Repository for Manuscript "A Novel Recurrent Neural Network Framework for Prediction and Treatment of Oncogenic Mutation Progression", Parthasarathy, Bhowmik.

# Reproducibility Checklist

To reproduce training results, download the desired TCGA Pan-Cancer Atlas data from cBioPortal. Then, import `networks.py` and run `run_model_stage_prediction` with the desired cancer class, number of epochs, omitted stages, and number of genes preprocessed (preprocessing is controlled by the `heatmap` flag).

This will train the network and produce ROC curves. 

To reproduce drug recommendation results, modify lines 44-46 of `predict_genes_drugs_official.py` to load in your model of choice, and then run `main.py`.

Overall, the specific RNN machine learning configuration used in this project contained an Embedding of length 256 (i.e. transforming each mutation into a float matrix of length 256), a bidirectional LSTM layer of length 64, and two Dense layers, which were activated with the Rectified Linear Unit (ReLU) and SoftMax, respectively. Training was performed using the Adam optimizer with a learning rate of 1e-4 over 200 epochs and a batch size of 16. Sequences were postfix-padded to the maximum length in the dataset using empty strings. No early stopping strategies were used, and convergence was analyzed by continued decrease in validation loss. No random seeds were used. These parameters were generated using hyperparameter sweeps checking for the highest validation accuracy. 

# Licenses/Citations
## TCGA 
"The Cancer Genome Atlas Program." National Cancer Institute at the National Institutes of
Health, National Institutes of Health, [www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga]. Accessed 5 Oct. 2020.

Cerami, Ethan, et al. "The CBio Cancer Genomics Portal: An Open Platform for Exploring Multidimensional Cancer Genomics Data: Figure 1." Cancer Discovery, vol. 2, no. 5, May 2012, pp. 401-04, doi:10.1158/2159-8290.CD-12-0095. Accessed 5 Oct. 2020.

Gao, J., et al. "Integrative Analysis of Complex Cancer Genomics and Clinical Profiles Using the CBioPortal." Science Signaling, vol. 6, no. 269, 26 Mar. 2013, p. pl1, doi:10.1126/scisignal.2004088. Accessed 5 Oct. 2020.

Gao, Jianjiong, et al., editors. cBioPortal for Cancer Genomics. Version 3.4.13, 2020, www.cbioportal.org/. Accessed 5 Oct. 2020.

## DrugBank
These DrugBank datasets are released under a [Creative Common’s Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

## BPS/IUPHAR Database

Armstrong JF, Faccenda E, Harding SD, Pawson AJ, Southan C, Sharman JL, Campo B, Cavanagh DR, Alexander SPH, Davenport AP, Spedding M, Davies JA; NC-IUPHAR. (2019) The IUPHAR/BPS Guide to PHARMACOLOGY in 2020: extending immunopharmacology content and introducing the IUPHAR/MMV Guide to MALARIA PHARMACOLOGY. Nucl. Acids Res. Volume 48, Issue D1, D1006-D1021. doi: 10.1093/nar/gkz951. [Full text](https://academic.oup.com/nar/article/48/D1/D1006/5613677). PMID: [31691834](https://pubmed.ncbi.nlm.nih.gov/31691834/)
