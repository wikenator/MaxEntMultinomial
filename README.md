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

## Example Commands

MaxEnt iterations, learning rate, and regularization coefficient can be modified in any maxent command.

### Run MaxEnt with no options
- unigram-only
- saves probabilities and weights to pickle files
- defaults to one fold using 80/20 train/test set split

`./maxent.py`

### Incorporate bigrams 
- uses unigrams and bigrams
- saves probabilities and weights to pickle files
- defaults to one fold using 80/20 train/test set split

`./maxent.py -b`

### Perform k-fold cross-validation
- use `-b` switch for bigram, otherwise defaults to unigrams
- train/test split is calculated as `(set size)/(# of folds)`
- below example calculates 10-fold cross-validation:

`./maxent.py [-b] -f 10`

### Read previously saved data
- use `-b` switch to use previously saved bigram data
- will retrain weights during MaxEnt classification
- if k-fold cross-validation was used to save data, use `-f #` switch again

`./maxent.py [-b] -l [-f #]`

### Read previously saved data and MaxEnt weights
- use `-b` switch to use previously saved bigram data
- if k-fold cross-validation was used to save data, use `-f #` switch again

`./maxent.py [-b] -l -R [-f #]`
