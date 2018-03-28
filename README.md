# MaxEntMultinomial
Multinomial Logistic Regression (Maximum Entropy) Classifier for classifying math word problems as related to a math subject.

## Usage
```
maxent.py [-h] [-b] [-t] [-r] [-l] [--steps STEPS] [--learn_rate LEARN_RATE]
          [--reg_coeff REG_COEFF]
```

`-h, --help` shows this help message and exits the program.

`-b, --use_bigrams` adds bigram features during maxent classifier training.

`-t, --use_trigrams` adds trigram features during maxent classifier training.

`-l, --load_pickle` loads data from pickle files. Data/probabilities will not be recalculated, even if bigram features are turned on/off.

`-r, --no_retrain` loads previously trained weights from pickle file.

`--steps STEPS`         Number of iterations for maxent gradient descent calculated during learning. Default: 1000

`--learn_rate LEARN_RATE` Learning rate to use during maxent learning. Default: 0.0005

`--reg_coeff REG_COEFF` Regularization coefficient to normalize maxent gradient descent during learning. Default: 0.001
