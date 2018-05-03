# MaxEntMultinomial
Multinomial Logistic Regression (Maximum Entropy) Classifier for classifying math word problems as related to a math subject.

## Usage
```
maxent.py [-h] [-b] [-t] [-d] [-p | -l] [-n] [-R]
          [ [-s | --steps] STEPS] [-f FOLDS]
	  [
            [ [-r | --learn_rate] LEARN_RATE]
	    [ [-c | --reg_coeff] REG_COEFF]
	    | [-g | --grid_search]
	  ]
```

`-h, --help` shows this help message and exits the program.

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

## Example Commands

MaxEnt iterations, learning rate, and regularization coefficient can be modified in any maxent command.

### Run MaxEnt with no options
- unigram only
- use `-p` switch to save probabilities and weights to pickle files, otherwise defaults to standard output
- defaults to one fold using 80/20 train/test set split

`./maxent.py [-p]`

### Incorporate bigrams 
- uses unigrams and bigrams
- defaults to one fold using 80/20 train/test set split

`./maxent.py -b`

### Incorporate dependency parse 
- use `-b` switch to use bigrams, otherwise defaults to unigram only
- defaults to one fold using 80/20 train/test set split

`./maxent.py [-b] -d`

### Perform k-fold cross-validation
- use `-b` switch for bigram, otherwise defaults to unigram only
- use `-d` switch for dependency parse
- train/test split is calculated as `(set size)/(# of folds)`
- below example calculates 10-fold cross-validation:

`./maxent.py [-b] [-d] -f 10`

### Read previously saved data
- use `-b` switch to use previously saved bigram data
- use `-d` switch to use previously saved dependency parse data
- will retrain weights during MaxEnt classification
- if k-fold cross-validation was used to save data, use `-f #` switch again

`./maxent.py [-b] [-d] -l [-f #]`

### Read previously saved data and MaxEnt weights
- use `-b` switch to use previously saved bigram data
- use `-d` switch to use previously saved dependency parse data
- if k-fold cross-validation was used to save data, use `-f #` switch again

`./maxent.py [-b] [-d] -l -R [-f #]`
