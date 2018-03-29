# MaxEntMultinomial
Multinomial Logistic Regression (Maximum Entropy) Classifier for classifying math word problems as related to a math subject.

## Usage
```
maxent.py [-h] [-b] [-t] [-l] [-n] [-R]
          [ [-s | --steps] STEPS] [ [-r | --learn_rate] LEARN_RATE]
          [ [-c | --reg_coeff] REG_COEFF] [-f FOLDS]
```

`-h, --help` shows this help message and exits the program.

`-b, --use_bigrams` adds bigram features during maxent classifier training.

`-t, --use_trigrams` adds trigram features during maxent classifier training.

`-l, --load_pickle` loads data from pickle files. Data/probabilities will not be recalculated, even if bigram features are turned on/off.

`-n, --naive` runs a simple Naive Bayes classifier before beginning MaxEnt calculations.

`-R, --no_retrain` loads previously trained weights from pickle file.

`-s STEPS, --steps STEPS` Number of iterations for maxent gradient descent calculated during learning. Default: 1000

`-r LEARN_RATE, --learn_rate LEARN_RATE` Learning rate to use during maxent learning. Default: 0.0005

`-c REG_COEFF, --reg_coeff REG_COEFF` Regularization coefficient to normalize maxent gradient descent during learning. Default: 0.001

`-f FOLDS, --folds FOLDS` Perform k-fold cross-validation. Larger k = less bias, more variance. Smaller k = more bias, less variance. Accuracy from each cross-validation will be averaged over all folds. Default: 1
