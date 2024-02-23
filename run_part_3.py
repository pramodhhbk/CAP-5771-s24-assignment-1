#######################################################################

import numpy as np
import utils as u
from numpy.typing import NDArray
from part_3_template_solution import Section3 as Part3

#######################################################################

# The argument `self` refers to a class, specifically class Part3, which is a shortcut for Section3

# Run this file from the command line or from VSCode
#   python run_part3.py


"""
A. Using the same classifier and hyperparameters as the one used at the end of 
   part 2.B.  Get the accuracies of the training/test set scores using the 
   top_k_accuracy score for k=1,2,3,4,5. Make a plot of k vs. score and comment 
   on the rate of accuracy change.  Do you think this metric is useful for this dataset?
"""


def part_A(
    self,
    X: NDArray[np.floating],
    y: NDArray[np.int32],
    Xtest: NDArray[np.floating],
    ytest: NDArray[np.int32],
):
    """ """
    answer, X, y, Xtest, ytest = self.partA(X, y, Xtest, ytest)
    return answer, X, y, Xtest, ytest


"""
B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed. 
   Also convert the 7s to 0s and 9s to 1s.

"""


def part_B(
    self,
    X: NDArray[np.floating],
    y: NDArray[np.int32],
    Xtest: NDArray[np.floating],
    ytest: NDArray[np.int32],
):
    answer, X, y, Xtest, ytest = self.partB(X, y, Xtest, ytest)
    return answer, X, y, Xtest, ytest


"""
C. Repeat part 1.C for this dataset but use a support vector machine (SVC in 
   sklearn. Do not use linearSVC.) Make sure to use a stratified cross-validation 
   strategy. In addition to regular accuracy also print out the mean/std of 
   the F1 score, precision, and recall. Is precision or recall higher? Explain. Finally, 
   train the classifier on all the training data and plot the confusion matrix.
"""


def part_C(
    self,
    X: NDArray[np.floating],
    y: NDArray[np.int32],
    Xtest: NDArray[np.floating],
    ytest: NDArray[np.int32],
):
    answer = self.partC(X, y, Xtest, ytest)
    return answer


"""
D. Repeat the same steps as part 3.C but apply a weighted loss function 
   (see the class_weights parameter). Print out the class weights, and 
   comment on the performance difference. Tip: compute_class_weight.
"""


def part_D(
    self,
    X: NDArray[np.floating],
    y: NDArray[np.int32],
    Xtest: NDArray[np.floating],
    ytest: NDArray[np.int32],
):
    answer = self.partD(X, y, Xtest, ytest)
    return answer


# ------------------------------------------------------------------
if __name__ == "__main__":
    """
    Run your code and produce all your results for your report. We will
    spot check the reports, and grade your code with automatic tools.
    """

    # Attention: the seed should never be changed. If it is, automatic grading
    # of the assignment could very well fail, and you'd lose points.
    # Make sure that all sklearn functions you use that require a seed have this
    # seed specified in the argument list, namely: `random_state=self.seed` if
    # you are inside the solution class.
    part3 = Part3(seed=42, frac_train=0.2)

    import utils as u

    Xtrain, ytrain, Xtest, ytest = u.prepare_data()

    answer3A, Xtrain, ytrain, Xtest, ytest = part_A(part3, Xtrain, ytrain, Xtest, ytest)
    answer3B, X_bin, y_bin, Xtest_bin, ytest_bin = part_B(
        part3, Xtrain, ytrain, Xtest, ytest
    )
    answer3C = part_C(part3, X_bin, y_bin, Xtest_bin, ytest_bin)
    answer3D = part_D(part3, X_bin, y_bin, Xtest_bin, ytest_bin)

    answer = {}
    answer["3A"] = answer3A
    answer["3B"] = answer3B
    answer["3C"] = answer3C
    answer["3D"] = answer3D
    
    u.save_dict("section3.pkl", answer)
