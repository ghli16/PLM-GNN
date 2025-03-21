## PLM-GNN：Classification of virulence factor based on dual-channel neural networks with pre-trained language models

This is the repository related to our manuscript Classification of virulence factor based on dual-channel neural networks with pre-trained language models, currently in submission at Interdisciplinary Sciences: Computational Life Sciences.

## Code
### Environment Requirement

The code has been tested running under Python 3.8.10. The required packages are as follows:
- numpy == 1.24.4
- scipy == 1.10.1
- pytorch == 1.10.0
- networkx == 3.1
- biopython == 1.81
- pandas == 1.5.3
- matplotlib == 3.5.0
- fair-esm == 2.0.0
- pytorch-geometric == 2.4.0
- pytorch-cuda == 11.3  

## Files

1. data: 
The training, validation, and testing sets used for multi-classification of virulence factors, as well as for classifying the 5 subcategories under Effector delivery systems.

2. PLM-GNN: 
           a.dataset.py：data processing module； 
           b.model.py: specific implementation of the PLM-GNN;
           c.utils.py: model utility class;

3. feature:
           a.feature.py：ProTrans Feature Generation Module; 
           b.gen_X.py: 3D Coordinate Generation Module.

## Train and Predict

 First, PDB files are used to generate 3D coordinates through genX.py in the feature folder. Then, ProTrans features are obtained using feature.py. Subsequently, these features, along with the fasta files, are utilized to train the model using train.sh. Finally, predict.py is used for making predictions.

