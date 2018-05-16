# MaxEntMultinomial
Multinomial Logistic Regression (Maximum Entropy) Classifier for classifying math word problems as related to a math subject.

## Usage
```
maxent.py [-h] [-u] [-b] [-t] [-d] [-p | -l] [-n] [-R]
          [ [-s | --steps] STEPS] [-f FOLDS] [--seed SEED]
	  [
            [ [-r | --learn_rate] LEARN_RATE]
	    [ [-c | --reg_coeff] REG_COEFF]
	    | [-g | --grid_search]
	  ]
```

`-h, --help` shows this help message and exits the program.

`-u, --use_unigrams` adds unigram features during maxent classifier training.

`-b, --use_bigrams` adds bigram features during maxent classifier training.

`-t, --use_trigrams` adds trigram features during maxent classifier training.

`-d, --dep_parse` adds dependency parsing features during maxent classifier training.

`-p, --save_pickle` saves data to pickle files. Cannot be used with `--load_pickle` option.

`-l, --load_pickle` loads data from pickle files. Data/probabilities will not be recalculated, even if bigram features are turned on/off. Cannot be used with `--save_pickle` option.

`-n, --naive` runs a simple Naive Bayes classifier before beginning MaxEnt calculations.

`-R, --no_retrain` loads previously trained weights from pickle file.

`-s STEPS, --steps STEPS` Number of iterations for maxent gradient descent calculated during learning. Default: 1000

`-r LEARN_RATE, --learn_rate LEARN_RATE` Learning rate to use during maxent learning. Default: 0.00007

`-c REG_COEFF, --reg_coeff REG_COEFF` Regularization coefficient to normalize maxent gradient descent during learning. Default: 0.5

`-g, --grid_search` Searches 2,025 learning rate and regulation coefficient combinations to find the best values during maxent learning. Uses the best trained values for the provided test set.

`-f FOLDS, --folds FOLDS` Perform k-fold cross-validation. Larger k = less bias, more variance. Smaller k = more bias, less variance. Accuracy from each cross-validation will be averaged over all folds. Default: 1

`--seed SEED` Set seed for random data shuffle. Use the same seed when saving and then subsequently loading pickle files. Default: 1

## Example Commands

MaxEnt iterations, learning rate, and regularization coefficient can be modified in any maxent command.

### Run MaxEnt with unigrams
- use `-p` switch to save probabilities and weights to pickle files, otherwise defaults to standard output
- defaults to one fold using 90/10 train/test set split

`./maxent.py [-p]`

### Run MaxEnt with bigrams 
- defaults to one fold using 90/10 train/test set split

`./maxent.py -b`

### Run MaxEnt with dependency parse 
- use `-u` switch to use unigrams or `-b` switch to use bigrams
- defaults to one fold using 80/20 train/test set split

`./maxent.py [-u] [-b] -d`

### Perform k-fold cross-validation
- use `-u` switch for unigrams
- use `-b` switch for bigrams
- use `-d` switch for dependency parse
- use `--seed` switch to choose seed value for random training data
- train/test split is calculated as `(set size)/(# of folds)`
- below example calculates 10-fold cross-validation:

`./maxent.py [-u] [-b] [-d] [--seed #] -f 10`

### Read previously saved data
- use `-u` switch to use previously saved unigram data
- use `-b` switch to use previously saved bigram data
- use `-d` switch to use previously saved dependency parse data
- will retrain weights during MaxEnt classification
- if k-fold cross-validation was used to save data, use `-f #` switch again
- if a unique seed was used during data save, use `--seed #` switch with same seed again

`./maxent.py [-u] [-b] [-d] -l [--seed #] [-f #]`

### Read previously saved data and MaxEnt weights
- use `-u` switch to use previously saved unigram data
- use `-b` switch to use previously saved bigram data
- use `-d` switch to use previously saved dependency parse data
- if k-fold cross-validation was used to save data, use `-f #` switch again
- if a unique seed was used during data save, use `--seed #` switch with same seed again

`./maxent.py [-u] [-b] [-d] -l -R [--seed #] [-f #]`
