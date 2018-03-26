# MaxEntMultinomial
Multinomial Logistic Regression (Maximum Entropy) Classifier for classifying math word problems as related to a math subject.

## Usage
```
maxent.py [--help | -h] [--use_bigrams | -b] [--load_pickle | -l]
```

`-h, --help` shows this help message and exits the program.

`-b, --use_bigrams` adds bigram features during maxent classifier training.

`-l, --load_pickle` loads data from pickle files. Data/probabilities will not be recalculated, even if bigram features are turned on/off.
