# DMLS2020-FinalProj


## R09922A01

### Extra changes
1. Apply Weighted Sampler
2. Standarize data before training

### Usage
Modify ```ModelPath``` and ```DataPath``` before training.

Command: python train.py [arg value]

Example: python train.py --bs 32

For more details please read ```argparser.py```

### Result

site | Data Size | Training Accuracy | Validation Accuracy
-----|-----------|-------------------|--------------------
  0  |   8261    |      98.83%       |        41.38%       
  1  |   9497    |      98.26%       |        49.08%
  2  |   9242    |      99.88%       |        64.36%
  3  |   8289    |      99.98%       |        80.46%
