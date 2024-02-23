#######################################################################
from numpy.typing import NDArray
import numpy as np
from typing import Any
from part_1_template_solution import Section1 as Part1
import utils as u


#######################################################################

# Use Python 10.x
# Run this file from the command line
#   python run_part1.py
# Will not run properly from VSCode, and I don't understand

# The argument `self` refers to a class, specifically class Part1, which is a shortcut for Section1

"""
A. We will start by ensuring that your python environment is configured correctly and 
   that you have all the required packages installed. For information about setting up 
   Python please consult the following link: https://www.anaconda.com/products/individual. 
   To test that your environment is set up correctly, import and run
   mnist_assignment_starter.py module. You may also want to run the file as a script.
"""


def part_A(self):
    answer = self.partA()
    return answer


"""
B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
   functions, as a data matrix 𝑋 consisting of only the digits 7 and 9. Make sure that 
   every element in the data matrix is a floating point number and scaled between 0 and 1. 
   Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
   and the maximum value of 𝑋 for both training and test sets.
"""


def part_B(self):
    answer, Xtrain, ytrain, Xtest, ytest = self.partB()
    return answer, Xtrain, ytrain, Xtest, ytest


"""
C. Train your first classifier using 𝑘-fold cross validation (see train_simple_classifier_with_cv 
   function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
   for the accuracy scores in each validation set in cross validation. Also print the mean and std 
   of the fit (or training) time.  (Be more specific about the output format)
"""


def part_C(self, X: NDArray[np.floating], y: NDArray[np.int32]):
    # scores, clf, cv = self.partC(X, y)
    answer = self.partC(X, y)
    return answer


"""
D. Repeat Part C with a random permutation 𝑘-fold cross-validator.
"""


def part_D(self, X: NDArray[np.floating], y: NDArray[np.int32]):
    answer = self.partD(X, y)
    # scores, clf, cv = self.partD(X, y, n_splits=5)
    # n_splits = 5
    # clf = DecisionTreeClassifier(random_state=self.seed)
    # cv = ShuffleSplit(n_splits=n_splits, random_state=self.seed)
    # scores = self.train_simple_classifier_with_cv(  # Not sure this is needed
    # X, y, cv=cv, clf=clf, n_splits=n_splits
    # )
    return answer


# Perhaps specify which functions to use?


"""
E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. Note that this may 
   take a long time (2–5 mins) to run. Do you notice anything about the mean and/or 
   standard deviation of the scores for each k?
"""


def part_E(self, X: NDArray[np.floating], y: NDArray[np.int32]):
    # Return dictionary (keys: n_split). Value: dict with keys 'scores', 'clf', 'cv'
    all_data = self.partE(X, y)
    return all_data


"""
F. Repeat part D with both logistic regression and support vector machine). Make sure 
   the train test splits are the same for both models when performing cross-validation. 
   Use ShuffleSplit for cross-validation. Which model has the highest accuracy on average? 
   Which model has the lowest variance on average? Which model is faster to train?
"""


def part_F(self, X: NDArray[np.floating], y: NDArray[np.int32]):
    # Return dictionary (keys: n_split). Value: dict with keys 'scores', 'clf', 'cv'
    all_data = self.partF(X, y)
    return all_data


"""
G. For the SVM classifier trained in part F, manually (or systematically, i.e., using grid search), 
   modify hyperparameters, and see if you can get a higher mean accuracy. Finally train the 
   classifier on all the training data and get an accuracy score on the test set. Print out the 
   training and testing accuracy and comment on how it relates to the mean accuracy when 
   performing cross validation. Is it higher, lower or about the same?
"""


def part_G(
    self,
    X: NDArray[np.floating],
    y: NDArray[np.int32],
    Xtest: NDArray[np.floating],
    ytest: NDArray[np.int32],
) -> dict[str, Any]:
    """
    Perform classification using the given classifier and cross validator.

    Parameters:
    - X: The test data.
    - y: The test labels.
    - n_splits: The number of splits for cross validation. Default is 5.

    Returns:
    - y_pred: The predicted labels for the test data.

    Note:
    This function is not fully implemented yet.
    """
    # Accuracy score on the test set. HOW TO DO THIS?
    # Count how many match between y_pred and y
    # clf, clf_gs = self.partG(X, y, Xtest, ytest)
    # print("clf= ", clf)
    # return clf, clf_gs
    answer = self.partG(X, y, Xtest, ytest)
    return answer


# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Run your code and produce all your results for your report. We will spot check the
    reports, and grade your code with automatic tools.
    """

    ################################################
    # In real code, read MNIST files and define Xtrain and xtest appropriately
    X = np.random.rand(120, 120)  # 100 samples, 100 features
    # Fill labels with 0 and 1 (mimic 7 and 9s)
    y = (X[:, :5].sum(axis=1) > 2.5).astype(int)
    n_train = 100
    Xtrain = X[0:n_train, :]
    Xtest = X[n_train:, :]
    ytrain = y[0:n_train]
    ytest = y[n_train:]
    X = Xtrain
    y = ytrain
    ##############################################

    # Attention: the seed should never be changed. If it is, automatic grading
    # of the assignment could very well fail, and you'd lose points.
    # Make sure that all sklearn functions you use that require a seed have this cle
    # seed specified in the argument list, namely: `random_state=self.seed` if
    # you are inside the solution class.
    part1 = Part1(seed=42, frac_train=0.2)
    #answer1B, X, y , Xtest, ytest = part_B(part1)
    # X and Y are Mnist datasets
    answer1A = part_A(part1)
    answer1B,X,y,Xtest,ytest = part_B(part1)
    answer1C = part_C(part1, X, y)
    answer1D = part_D(part1, X, y)
    answer1E = part_E(part1, X, y)
    answer1F = part_F(part1, X, y)

    # clf,  #: Type[BaseEstimator],  # Estimator class instance
    # cv,  #: Type[BaseCrossValidator],  # Cross Validator class instsance
    answer1G = part_G(part1, X, y, Xtest, ytest)

    answer = {}
    answer["1A"] = answer1A
    answer["1B"] = answer1B
    answer["1C"] = answer1C
    answer["1D"] = answer1D
    answer["1E"] = answer1E
    answer["1F"] = answer1F
    answer["1G"] = answer1G
    #print(answer)
    #print(answer["1G"])
    #print(answer["1F"])

    u.save_dict("section1.pkl", answer)
