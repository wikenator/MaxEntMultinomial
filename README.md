# MaxEntMultinomial
Multinomial Logistic Regression (Maximum Entropy) Classifier for classifying math word problems as related to a math subject.

## Usage
```
maxent.py [--help | -h] [--use_bigrams | -b] [--load_pickle | -l]
```

`-h, --help` shows this help message and exits the program.

`-b, --use_bigrams` adds bigram features during maxent classifier training.

`-l, --load_pickle` loads data from pickle files. Data/probabilities will not be recalculated, even if bigram features are turned on/off.

`--steps STEPS`         Number of iterations for maxent gradient descent calculated during learning. Default: 400

`--learn_rate LEARN_RATE` Learning rate to use during maxent learning. Default: 0.0005

`--reg_coeff REG_COEFF` Regularization coefficient to normalize maxent gradient descent during learning. Default: 0.001
