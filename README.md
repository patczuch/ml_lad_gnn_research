# LAD-GNN Research

## Label Attentive Distillation for Graph Neural Networks

### Authors
- Patryk Czuchnowski
- Michał Pędrak
- Andrzej Wacławik

## 1. Introduction

This project investigates the **LAD** mechanism introduced in the paper *"Label Attentive Distillation for GNN-Based Graph Classification"* presented at AAAI-24. Our goal was to independently evaluate and verify the effectiveness of the LAD approach across multiple GNN architectures and diverse graph classification datasets.

The original paper claims that conventional GNNs suffer from an "embedding misalignment" - problem, where node embeddings generated without considering graph-level label information lead to suboptimal graph representations for classification tasks. LAD-GNN proposes a teacher-student distillation framework to address this issue.

## 2. What is the LAD Mechanism?

The **Label Attentive Distillation** mechanism is a two-phase training approach designed to improve GNN performance on graph classification tasks:

### Phase 1: Label-Attentive Teacher Training
- A **label-attentive encoder** encodes ground-truth labels into label embeddings
- These label embeddings are combined with node embeddings from the GNN backbone using an attention mechanism (similar to Transformer architecture)
- The resulting "ideal embedding" fuses global label information with local node features
- The teacher model is trained to minimize classification loss using these enhanced embeddings

Implementation of the attention mechanism used in the Teacher model from `models/lad_base.py` is presented below:

```python
class linear_attention(nn.Module):
    def __init__(self, in_dim):
        super(linear_attention, self).__init__()
        self.layerQ = nn.Linear(in_dim, in_dim) # query from label embeddings
        self.layerK = nn.Linear(in_dim, in_dim) # key from node embeddings
        self.layerV = nn.Linear(in_dim, in_dim) # value from node embeddings
        self.initialize()

    def initialize(self): # initialize parameters
        self.layerQ.reset_parameters()
        self.layerK.reset_parameters()
        self.layerV.reset_parameters()

    def forward(self, node_emb, label_emb, tau=0.5):
        Q = self.layerQ(label_emb)
        K = self.layerK(node_emb)
        V = self.layerV(node_emb)
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) # [n_labels, n_nodes], scaled dot-product
        attention_weight = F.softmax(attention_score * tau, dim=1) # [n_labels, n_nodes], softmax over nodes
        z = torch.matmul(attention_weight, V) # [n_labels, in_dim], weighted sum
        return z
```

### Phase 2: Distillation-based Student Learning
- A student GNN model learns to generate class-friendly node embeddings by distilling knowledge from the teacher
- The student shares the classification head with the teacher but does not use label information during inference
- Training minimizes both classification loss and a distillation loss (MSE between teacher and student embeddings):

```
L = L_cls + λ · L_dis
```

where L is total loss value, L_cls is classification loss, λ is chosen ratio and L_dis is distillation loss.
Below is code we are using for achieving this from `main.py`

```python
  out, st_map = model(data.x, data.edge_index, data.batch)  # student model
  _, te_map = teacher_model(data.x, labs, data.edge_index, data.batch) # teacher model
  loss_distill = mse_loss(te_map, st_map) 
  loss_classification = nll_loss(out, data.y.view(-1))
  loss = loss_classification + args.alpha * loss_distill
```

During inference, only the student model is used, ensuring no information leakage from labels.

## 3. Tested GNN Architectures

We evaluated the LAD mechanism on **4 commonly used GNN backbones**, source code for them is in the `models` folder:

| Architecture | Description |
|--------------|-------------|
| **GCN** | Graph Convolutional Network - pioneering spectral convolution approach |
| **GAT** | Graph Attention Network - uses attention mechanism for neighbor aggregation |
| **GIN** | Graph Isomorphism Network - maximally powerful in the Weisfeiler-Lehman test |
| **GraphSAGE** | Sample and Aggregate - inductive representation learning on large graphs |

Each backbone was tested in two configurations:
- **Pure**: Standard training without LAD
- **Student + Teacher**: Training with LAD distillation with a label-attentive teacher

The backbone selection logic ensures fair comparison by plugging different GNNs into the same framework structure as presented below (code fragment from `models/lad_base.py`):

```python
class STnet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, gnn, nlayers=2, gat_heads=4, dropout=0.5, with_bn=True, with_bias=True):
        
        super(STnet, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass

        if gnn == "GCN":
            self.gnn_model = GCN(nfeat, nhid, nhid, nlayers, dropout, with_bias, with_bn)
        elif gnn == "GAT":
            self.gnn_model = GAT(nfeat, nhid, nhid, nlayers, gat_heads, 1, dropout, with_bn)
        elif gnn == "GIN":
            self.gnn_model = GIN(nfeat, nhid, nhid, nlayers, dropout, with_bias, with_bn)
        elif gnn == "GraphSAGE":
            self.gnn_model = GraphSAGE(nfeat, nhid, nhid, nlayers, dropout, with_bias, with_bn)
        else:
            raise Exception("Invalid GNN type!")

        self.classifier = Classifier(nhid, nclass)
```            

## 4. Datasets

We selected datasets not tested in the original paper, and not only from chemical domain to evaluate LAD's generalizability:

### Molecular and chemical datasets

| Dataset | Graphs | Avg. Nodes | Classes | Description |
|---------|--------|------------|---------|-------------|
| **MUTAG** | 188 | ~18 | 2 | Mutagenic compounds - predicts mutagenicity on Salmonella typhimurium |
| **PROTEINS** | 1,113 | ~39 | 2 | Protein structures - enzyme vs non-enzyme classification |
| **ogbg-molhiv** | 41,127 | ~25 | 2 | HIV virus replication inhibition prediction from Open Graph Benchmark |

### Social network datasets

| Dataset | Graphs | Avg. Nodes | Classes | Description |
|---------|--------|------------|---------|-------------|
| **COLLAB** | 5,000 | ~74 | 3 | Scientific collaboration networks - classifies researcher's field |
| **IMDB-BINARY** | 1,000 | ~20 | 2 | Movie collaboration networks - genre classification (Action/Romance) |
| **IMDB-MULTI** | 1,500 | ~13 | 3 | Movie collaboration networks - multi-class genre classification |
| **REDDIT-BINARY** | 2,000 | ~430 | 2 | Reddit thread graphs - discussion vs Q&A community classification |

## 5. Testing Methodology

### Experimental Setup

Each combination of **(dataset, backbone, training mode)** was run **10 times** with different random seeds to ensure statistical reliability. This is implemented in `run_training_series.py`:

```python
seeds = [rng.randint(1, 2**31 - 1) for _ in range(args.runs)]  # args.runs = 10

for run_idx, seed in enumerate(seeds, 1):
    for backbone in args.backbones:   # ['GAT', 'GCN', 'GIN', 'GraphSAGE']
        for mode in args.modes:       # ['P', 'T', 'S']
            # ...
            # launch training with consistent seed
            # ...
```

### Metrics Calculated

We compute the following metrics for comprehensive evaluation (see `scripts_gen_tables\gen_table.py`):

| Metric | Description |
|--------|-------------|
| **Accuracy** | Classification accuracy on test set |
| **AUC** | Area Under ROC Curve (for binary classification datasets) |
| **F1** | Weighted F1 score |
| **ECE** | Expected Calibration Error - measures prediction confidence calibration |
| **Brier** | Brier score - measures probabilistic prediction accuracy |

Results are aggregated as **mean ± standard deviation** across the 10 runs.

### Data Splitting

Following the original paper, datasets were split using **10-fold cross-validation** with an 80/10/10 train/validation/test protocol.

### Hyperparameters

Hyperparameters for each dataset can be found described in `run_training_series.py`

## 6. Results

### COLLAB
| Model    | Accuracy        | F1              | ECE             | Brier           |
|:---------|:----------------|:----------------|:----------------|:----------------|
| GAT      | 0.6052 ± 0.0336 | 0.5511 ± 0.0369 | 0.0714 ± 0.0154 | 0.5265 ± 0.0282 |
| GAT+LAD  | 0.5692 ± 0.0335 | 0.5195 ± 0.0372 | 0.1610 ± 0.0561 | 0.5857 ± 0.0544 |
| GCN      | 0.7042 ± 0.0167 | 0.6982 ± 0.0166 | 0.0388 ± 0.0084 | 0.4125 ± 0.0114 |
| GCN+LAD  | 0.7212 ± 0.0194 | 0.7157 ± 0.0181 | 0.0383 ± 0.0095 | 0.3930 ± 0.0192 |
| GIN      | 0.7050 ± 0.0212 | 0.6967 ± 0.0209 | 0.0411 ± 0.0062 | 0.4177 ± 0.0187 |
| GIN+LAD  | 0.7236 ± 0.0231 | 0.7201 ± 0.0225 | 0.0379 ± 0.0053 | 0.3908 ± 0.0159 |
| SAGE     | 0.6888 ± 0.0244 | 0.6859 ± 0.0242 | 0.0411 ± 0.0055 | 0.4360 ± 0.0188 |
| SAGE+LAD | 0.7216 ± 0.0184 | 0.7215 ± 0.0188 | 0.0394 ± 0.0080 | 0.3978 ± 0.0242 |
- **LAD impact**
  - **Positive** for: SAGE (+3.28pp), GIN (+1.86pp) and GCN (+1.7pp) 
  - **Negative** for: GAT (-3.6pp)
- **Best model**: GIN+LAD (72.36% accuracy)
- **Observations**: 
  - LAD improved the simpler aggregation-based models (GCN, GIN, SAGE)
  - GAT's attention mechanism may conflict with LAD's label-attentive encoder, leading to degraded performance
  - ECE values remained stable across LAD variants, indicating the mechanism doesn't harm calibration

### IMDB-BINARY
| Model    | Accuracy        | AUC             | F1              | ECE             | Brier           |
|:---------|:----------------|:----------------|:----------------|:----------------|:----------------|
| GAT      | 0.5580 ± 0.0365 | 0.5762 ± 0.0379 | 0.5409 ± 0.0397 | 0.0102 ± 0.0118 | 0.4923 ± 0.0075 |
| GAT+LAD  | 0.5290 ± 0.0687 | 0.5290 ± 0.0821 | 0.4608 ± 0.1099 | 0.0656 ± 0.0517 | 0.5071 ± 0.0243 |
| GCN      | 0.6770 ± 0.0483 | 0.7274 ± 0.0575 | 0.6742 ± 0.0478 | 0.0327 ± 0.0131 | 0.4466 ± 0.0182 |
| GCN+LAD  | 0.6940 ± 0.0542 | 0.7180 ± 0.0703 | 0.6907 ± 0.0537 | 0.0263 ± 0.0118 | 0.4272 ± 0.0396 |
| GIN      | 0.6450 ± 0.0525 | 0.7097 ± 0.0736 | 0.6329 ± 0.0601 | 0.0318 ± 0.0181 | 0.4531 ± 0.0220 |
| GIN+LAD  | 0.7150 ± 0.0564 | 0.7540 ± 0.0551 | 0.7136 ± 0.0560 | 0.0260 ± 0.0164 | 0.4114 ± 0.0382 |
| SAGE     | 0.6470 ± 0.0380 | 0.6656 ± 0.0403 | 0.6433 ± 0.0398 | 0.0184 ± 0.0100 | 0.4617 ± 0.0156 |
| SAGE+LAD | 0.5890 ± 0.0448 | 0.6521 ± 0.0461 | 0.5531 ± 0.0622 | 0.0393 ± 0.0192 | 0.4879 ± 0.0195 |

- **LAD impact**
  - **positive** for: GIN (+7pp) and GCN (+1.7pp) 
  - **negative** for: SAGE (-5.8pp) and GAT (-2.9pp) 
- **Best model**: GIN+LAD (71.5% accuracy, 0.75 AUC)
- **Observations**:
  - This dataset shows the most variance in LAD effectiveness across architectures
  - GIN benefits significantly, likely because its injective aggregation function pairs well with the class-specific knowledge from LAD
  - SAGE's performance degradation suggests that the sampling-based approach may not synergize well with distillation on smaller datasets
  - The high standard deviations indicate sensitivity to random initialization

### IMDB-MULTI
| Model    | Accuracy        | F1              | ECE             | Brier           |
|:---------|:----------------|:----------------|:----------------|:----------------|
| GAT      | 0.4327 ± 0.0221 | 0.3887 ± 0.0292 | 0.0091 ± 0.0072 | 0.6495 ± 0.0093 |
| GAT+LAD  | 0.4280 ± 0.0286 | 0.3854 ± 0.0375 | 0.0125 ± 0.0140 | 0.6505 ± 0.0056 |
| GCN      | 0.4633 ± 0.0472 | 0.4475 ± 0.0496 | 0.0329 ± 0.0148 | 0.6520 ± 0.0131 |
| GCN+LAD  | 0.4680 ± 0.0490 | 0.4590 ± 0.0516 | 0.0192 ± 0.0119 | 0.6389 ± 0.0143 |
| GIN      | 0.4720 ± 0.0312 | 0.4525 ± 0.0341 | 0.0384 ± 0.0122 | 0.6381 ± 0.0148 |
| GIN+LAD  | 0.4853 ± 0.0231 | 0.4761 ± 0.0244 | 0.0139 ± 0.0086 | 0.6294 ± 0.0117 |
| SAGE     | 0.4213 ± 0.0388 | 0.3783 ± 0.0365 | 0.0155 ± 0.0097 | 0.6479 ± 0.0125 |
| SAGE+LAD | 0.4267 ± 0.0278 | 0.3918 ± 0.0300 | 0.0182 ± 0.0067 | 0.6469 ± 0.0101 |
- **LAD impact**: Marginal **differences** across all models (-0.5pp to +1.5pp)
- **Best model**: GIN+LAD (48.53% accuracy)
- **Observations**:
  - Overall low accuracy across all models suggests this is a challenging dataset
  - LAD provides consistent but modest improvements
  - The 3-class setting with limited data (1,500 graphs) may not provide enough signal for effective distillation
  - ECE improvements with LAD suggest better-calibrated predictions despite similar accuracy

### MUTAG
| Model    | Accuracy        | AUC             | F1              | ECE             | Brier           |
|:---------|:----------------|:----------------|:----------------|:----------------|:----------------|
| GAT      | 0.7737 ± 0.1191 | 0.8537 ± 0.1025 | 0.7392 ± 0.1452 | 0.0610 ± 0.0532 | 0.3268 ± 0.1012 |
| GAT+LAD  | 0.7316 ± 0.1346 | 0.7711 ± 0.1308 | 0.6479 ± 0.1813 | 0.0476 ± 0.0402 | 0.3578 ± 0.1048 |
| GCN      | 0.8474 ± 0.0677 | 0.8978 ± 0.0863 | 0.8442 ± 0.0741 | 0.0676 ± 0.0261 | 0.2308 ± 0.0797 |
| GCN+LAD  | 0.8579 ± 0.0499 | 0.9239 ± 0.0439 | 0.8572 ± 0.0543 | 0.0563 ± 0.0270 | 0.2401 ± 0.0419 |
| GIN      | 0.8579 ± 0.1383 | 0.9252 ± 0.0873 | 0.8583 ± 0.1396 | 0.0783 ± 0.0708 | 0.2243 ± 0.1278 |
| GIN+LAD  | 0.8263 ± 0.0787 | 0.9144 ± 0.0813 | 0.8023 ± 0.0986 | 0.0538 ± 0.0368 | 0.2418 ± 0.0966 |
| SAGE     | 0.8053 ± 0.0861 | 0.8567 ± 0.0662 | 0.7782 ± 0.1128 | 0.0621 ± 0.0439 | 0.3124 ± 0.0629 |
| SAGE+LAD | 0.7526 ± 0.1165 | 0.8373 ± 0.0821 | 0.7113 ± 0.1500 | 0.0372 ± 0.0304 | 0.3539 ± 0.1210 |
- **LAD impact**
  - **positive** for: GCN (+1.05pp)
  - **negative** for SAGE (-5.27pp), GAT (-4.21pp) and GIN (-3.16pp)
- **Best model**: GIN (85.79% accuracy) and GCN+LAD (85.79% accuracy, but higher AUC at 0.92)
- **Observations**:
  - MUTAG is the smallest dataset (188 graphs), which may limit LAD's effectiveness
  - High standard deviations indicate high variance due to small dataset size
  - Interestingly, AUC improvements with LAD are more consistent than accuracy improvements

### PROTEINS
| Model    | Accuracy        | AUC             | F1              | ECE             | Brier           |
|:---------|:----------------|:----------------|:----------------|:----------------|:----------------|
| GAT      | 0.6964 ± 0.0362 | 0.6979 ± 0.0519 | 0.6744 ± 0.0401 | 0.0467 ± 0.0243 | 0.4386 ± 0.0169 |
| GAT+LAD  | 0.7196 ± 0.0260 | 0.7733 ± 0.0297 | 0.7208 ± 0.0252 | 0.0490 ± 0.0147 | 0.3971 ± 0.0303 |
| GCN      | 0.7161 ± 0.0447 | 0.7352 ± 0.0484 | 0.7023 ± 0.0601 | 0.0359 ± 0.0147 | 0.4111 ± 0.0232 |
| GCN+LAD  | 0.7402 ± 0.0484 | 0.8113 ± 0.0474 | 0.7399 ± 0.0492 | 0.0416 ± 0.0145 | 0.3606 ± 0.0495 |
| GIN      | 0.7232 ± 0.0514 | 0.7258 ± 0.0570 | 0.7125 ± 0.0600 | 0.0452 ± 0.0160 | 0.4187 ± 0.0284 |
| GIN+LAD  | 0.7411 ± 0.0455 | 0.7875 ± 0.0575 | 0.7358 ± 0.0466 | 0.0360 ± 0.0173 | 0.3687 ± 0.0407 |
| SAGE     | 0.7018 ± 0.0430 | 0.7047 ± 0.0441 | 0.6873 ± 0.0532 | 0.0514 ± 0.0237 | 0.4242 ± 0.0266 |
| SAGE+LAD | 0.7652 ± 0.0309 | 0.8125 ± 0.0307 | 0.7546 ± 0.0364 | 0.0362 ± 0.0139 | 0.3484 ± 0.0268 |
- **LAD impact**: Consistently **positive** across all architectures (+2.3pp to +6.34pp)
- **Best model**: SAGE+LAD (76.52% accuracy, 0.81 AUC)
- **Observations**:
  - This is the only dataset where LAD improved all four backbone architectures
  - SAGE benefits the most (+6.34pp accuracy), suggesting LAD helps compensate for SAGE's simpler aggregation
  - Brier scores improved significantly with LAD, indicating better probabilistic predictions
  - The protein structure classification task seems particularly well-suited to the label-attentive approach

### REDDIT-BINARY
| Model    | Accuracy        | AUC             | F1              | ECE             | Brier           |
|:---------|:----------------|:----------------|:----------------|:----------------|:----------------|
| GAT      | 0.7460 ± 0.0343 | 0.7898 ± 0.0412 | 0.7438 ± 0.0343 | 0.0284 ± 0.0186 | 0.3698 ± 0.0312 |
| GAT+LAD  | 0.7585 ± 0.0534 | 0.8491 ± 0.0534 | 0.7554 ± 0.0538 | 0.0482 ± 0.0177 | 0.3287 ± 0.0514 |
| GCN      | 0.7605 ± 0.0344 | 0.8493 ± 0.0302 | 0.7584 ± 0.0359 | 0.0555 ± 0.0118 | 0.3352 ± 0.0342 |
| GCN+LAD  | 0.8460 ± 0.0328 | 0.9251 ± 0.0183 | 0.8457 ± 0.0329 | 0.0248 ± 0.0049 | 0.2268 ± 0.0270 |
| GIN      | 0.8110 ± 0.0471 | 0.8947 ± 0.0368 | 0.8108 ± 0.0471 | 0.0457 ± 0.0143 | 0.2827 ± 0.0511 |
| GIN+LAD  | 0.8385 ± 0.0247 | 0.9183 ± 0.0166 | 0.8379 ± 0.0250 | 0.0292 ± 0.0078 | 0.2278 ± 0.0265 |
| SAGE     | 0.7745 ± 0.0277 | 0.8449 ± 0.0272 | 0.7732 ± 0.0285 | 0.0303 ± 0.0094 | 0.3213 ± 0.0273 |
| SAGE+LAD | 0.8550 ± 0.0486 | 0.9369 ± 0.0226 | 0.8529 ± 0.0517 | 0.0365 ± 0.0132 | 0.2112 ± 0.0503 |
- **LAD impact**: **Strong positive** across all architectures (+1.25pp to +8.55pp)
- **Best model**: SAGE+LAD (85.5% accuracy, 0.94 AUC)
- **Observations**:
  - GCN shows the largest improvement (+8.55% accuracy), demonstrating LAD's potential on larger graphs
  - This dataset has the largest average graph size (~430 nodes), suggesting LAD scales well
  - All LAD variants show improved Brier scores, indicating better uncertainty estimation
  - The binary classification on distinct community types (discussion vs Q&A) provides clear class signals for LAD

### ogbg-molhiv
| Model    | Accuracy        | AUC             | F1              | ECE             | Brier           |
|:---------|:----------------|:----------------|:----------------|:----------------|:----------------|
| GAT      | 0.9651 ± 0.0022 | 0.6562 ± 0.0229 | 0.9480 ± 0.0033 | 0.0082 ± 0.0040 | 0.0666 ± 0.0041 |
| GAT+LAD  | 0.9651 ± 0.0023 | 0.6825 ± 0.0311 | 0.9481 ± 0.0034 | 0.0069 ± 0.0015 | 0.0644 ± 0.0039 |
| GCN      | 0.9658 ± 0.0022 | 0.7014 ± 0.0284 | 0.9509 ± 0.0026 | 0.0068 ± 0.0014 | 0.0632 ± 0.0038 |
| GCN+LAD  | 0.9666 ± 0.0021 | 0.7120 ± 0.0369 | 0.9532 ± 0.0039 | 0.0080 ± 0.0015 | 0.0604 ± 0.0036 |
| GIN      | 0.9660 ± 0.0023 | 0.7063 ± 0.0274 | 0.9522 ± 0.0032 | 0.0065 ± 0.0016 | 0.0627 ± 0.0035 |
| GIN+LAD  | 0.9674 ± 0.0028 | 0.7134 ± 0.0491 | 0.9559 ± 0.0044 | 0.0082 ± 0.0010 | 0.0600 ± 0.0044 |
| SAGE     | 0.9653 ± 0.0024 | 0.6896 ± 0.0161 | 0.9483 ± 0.0035 | 0.0061 ± 0.0014 | 0.0646 ± 0.0041 |
| SAGE+LAD | 0.9667 ± 0.0026 | 0.7263 ± 0.0236 | 0.9548 ± 0.0049 | 0.0076 ± 0.0019 | 0.0610 ± 0.0044 |
- **LAD impact**: Consistently **positive** but modest (+0.08pp to +0.14pp accuracy, +0.01 to 0.037 AUC)
- **Best model**: GIN+LAD (96.74% accuracy, 0.71 AUC)
- **Observations**:
  - High baseline accuracy (~96.5%) due to class imbalance (most molecules are non-HIV inhibitors)
  - AUC is the more meaningful metric here, where LAD provides consistent 1-3% improvements
  - SAGE+LAD shows the best AUC improvement (+0.037), suggesting LAD helps with the minority class
  - The large dataset size (41k graphs) provides sufficient data for effective teacher training

## 7. Analysis of Model Rankings

### Summary of LAD impact per architecture
| Architecture | Datasets Improved | Average change in accuracy |
|--------------|-------------------|----------------------------|
| **GCN+LAD**  | 7/7               | +2.89pp                    |
| **SAGE+LAD** | 5/7               | +1.94pp                    |
| **GIN+LAD**  | 5/7               | +1.68pp                    |
| **GAT+LAD**  | 3/7               | -0.97pp                    |

### Mean position in ranking of each model per metric

| Model    |   AUC |   Accuracy |   Brier |   ECE |   F1 |  All |
|:---------|------:|-----------:|--------:|------:|-----:|-----:|
| GIN+LAD  |  2.00 |       1.86 |    2.00 |  3.14 | 2.14 | 2.23 |
| GCN+LAD  |  2.00 |       2.57 |    2.29 |  3.86 | 2.29 | 2.60 |
| SAGE+LAD |  2.57 |       3.71 |    3.57 |  4.29 | 3.29 | 3.49 |
| GIN      |  3.00 |       3.43 |    3.71 |  5.57 | 3.86 | 3.91 |
| GCN      |  3.29 |       4.57 |    4.86 |  5.00 | 4.43 | 4.43 |
| SAGE     |  4.57 |       5.86 |    5.71 |  4.43 | 5.86 | 5.29 |
| GAT      |  5.57 |       6.86 |    7.14 |  4.29 | 7.14 | 6.20 |
| GAT+LAD  |  5.00 |       7.14 |    6.71 |  5.43 | 7.00 | 6.26 |

Observations:
- **LAD-enhanced models are at the top positions**: GIN+LAD (2.23), GCN+LAD (2.60), and SAGE+LAD (3.49) occupy the top 3 positions, demonstrating the general effectiveness of the LAD mechanism.
- **GIN benefits most from LAD**: GIN+LAD achieves the best mean ranking (2.23), with particularly strong performance in Accuracy (1.86) and AUC (2.00). The GIN architecture's may make it particularly receptive to class-specific knowledge distillation.
- **GAT is the exception**: GAT+LAD (6.26) performs worse than pure GAT (6.20), making it the only architecture where LAD has a net negative effect. This is likely because:
   - GAT already uses attention mechanisms for neighbor aggregation
   - The additional label-attentive encoder may create competing attention signals
   - The model may become over-parameterized relative to dataset sizes
- **Brier score improvements are consistent**: LAD models consistently rank higher in Brier scores, indicating better probabilistic predictions overall.
- **ECE rankings are inconsistent**: Unlike other metrics, ECE (Expected Calibration Error) doesn't show a clear pattern favoring LAD models. This suggests that while LAD improves discriminative performance, its effect on calibration is dataset-dependent.



### When to use LAD?

- **Medium to large datasets**: LAD requires sufficient data to train an effective teacher model (e.g., MUTAG results).
- **Binary classification tasks**: The mechanism shows more consistent improvements on 2-class problems. We have also seen some improvements, albeit smaller on multi-class datasets
- **Graphs with clear class-distinguishing features**: Datasets like REDDIT-BINARY and PROTEINS where classes have distinct structural patterns
- **Use GCN, GIN, or GraphSAGE backbones**: These architectures consistently benefit from LAD. We don't recommend using GAT
- **When probabilistic calibration matters**: LAD generally improves Brier scores

## 8. Key Findings

1. **LAD mechanism generally works**: Across 7 datasets and 4 architectures (28 configurations), LAD improved performance in 20 cases (71.4%) and degraded it in 8 cases (28.6%).

2. **Architecture matters significantly**: The effectiveness of LAD is highly dependent on the GNN backbone:
   - GCN shows the most consistent improvements (7/7 datasets)
   - GIN and SAGE benefit in most cases (5/7 datasets each)
   - GAT frequently suffers from LAD (4/7 datasets show degradation)

3. **Dataset characteristics influence results**:
   - Larger datasets benefit more from LAD
   - Binary classification tasks show more consistent improvements
   - Social network datasets (COLLAB, REDDIT-BINARY) respond well to LAD

4. **The original paper's claims are mostly validated**:
   - We confirm that LAD can improve GNN performance on graph classification, although the mechanism is not universally beneficial across all cases.


## References

- Hong, X., Li, W., Wang, C., Lin, M., & Lu, S. (2024). Label Attentive Distillation for GNN-Based Graph Classification. *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-24)*.
- Original implementation: https://github.com/XiaobinHong/LAD-GNN


