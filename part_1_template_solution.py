# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u


# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        answer = u.starter_code()
        return answer

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

    def partB(
        self,
    ):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest) 

        answer = {}

        # Enter your code and fill the `answer` dictionary

        answer["length_Xtrain"] = Xtrain.shape[0]  # Number of samples
        answer["length_Xtest"] = Xtest.shape[0]
        answer["length_ytrain"] = ytrain.shape[0]
        answer["length_ytest"] = ytest.shape[0]
        answer["max_Xtrain"] = np.max(Xtrain)
        answer["max_Xtest"] = np.max(Xtest)
        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        """X, y, Xtest, ytest = u.prepare_data()
        X, y = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)"""
        # Enter your code and fill the `answer` dictionary
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = KFold(n_splits=5,random_state=self.seed,shuffle=True)
       
        scores = u.train_simple_classifier_with_cv(Xtrain=X,ytrain=y,clf=clf,cv=cv)
        
        score_dict={}
        for key,array in scores.items():
            if(key=='fit_time'):
                score_dict['mean_fit_time'] = array.mean()
                score_dict['std_fit_time'] = array.std()
            if(key=='test_score'):
                score_dict['mean_accuracy'] = array.mean()
                score_dict['std_accuracy'] = array.std()
        
        answer = {}
        answer["clf"] = clf  # the estimator (classifier instance)
        answer["cv"] = cv  # the cross validator instance
        # the dictionary with the scores  (a dictionary with
        # keys: 'mean_fit_time', 'std_fit_time', 'mean_accuracy', 'std_accuracy'.
        answer["scores"] = score_dict
        return answer

    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        """X, y, Xtest, ytest = u.prepare_data()
        X, y = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)"""
        # Enter your code and fill the `answer` dictionary
        scores = u.train_simple_classifier_with_cv(Xtrain=X,ytrain=y,clf=DecisionTreeClassifier(random_state=self.seed),cv=ShuffleSplit(n_splits=5,random_state=self.seed))
        score_dict={}
        for key,array in scores.items():
            if(key=='fit_time'):
                score_dict['mean_fit_time'] = array.mean()
                score_dict['std_fit_time'] = array.std()
            if(key=='test_score'):
                score_dict['mean_accuracy'] = array.mean()
                score_dict['std_accuracy'] = array.std()

        # Answer: same structure as partC, except for the key 'explain_kfold_vs_shuffle_split'

        answer = {}
        answer["clf"] = DecisionTreeClassifier(random_state=self.seed)
        answer["cv"] = ShuffleSplit(n_splits=5,random_state=self.seed)
        answer["scores"] = score_dict
        answer["explain_kfold_vs_shuffle_split"] = " K-fold cross-validation divides the dataset into k subsets for training and validation, while shuffle-split randomly shuffles and splits the dataset into training and validation sets multiple times"
        return answer

    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Answer: built on the structure of partC
        # `answer` is a dictionary with keys set to each split, in this case: 2, 5, 8, 16
        # Therefore, `answer[k]` is a dictionary with keys: 'scores', 'cv', 'clf`
        k =[2,5,8,16]
        answer={}
        """ X, y, Xtest, ytest = u.prepare_data()
        X, y = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)"""
        for k_value in k:
            # Calculate scores using train_simple_classifier_with_cv function
            scores = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=DecisionTreeClassifier(random_state=self.seed), cv=ShuffleSplit(n_splits=k_value,random_state=self.seed))
            
            # Initialize dictionary for storing scores
            score_dict = {}
            
            # Calculate mean and standard deviation of fit time
            # score_dict['mean_fit_time'] = np.mean(scores['fit_time'])
            # score_dict['std_fit_time'] = np.std(scores['fit_time'])
            
            # Calculate mean and standard deviation of accuracy
            
            score_dict['mean_accuracy'] = np.mean(scores['test_score'])
            score_dict['std_accuracy'] = np.std(scores['test_score'])
            
            # Populate answer dictionary
            answer[k_value] = {'scores': score_dict, 'cv': ShuffleSplit(n_splits=k_value,random_state=self.seed), 'clf': DecisionTreeClassifier(random_state=self.seed)}


        # Enter your code, construct the `answer` dictionary, and return it.

        return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ) -> dict[str, Any]:
        """ """
        """X, y, Xtest, ytest = u.prepare_data()
        X, y = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)"""
        answer = {}
        answer_D = self.partD(X,y)
        
        # Calculate scores using train_simple_classifier_with_cv function
        scores_rf = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=RandomForestClassifier(random_state=self.seed), cv=answer_D["cv"])
        #scores_dt = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=DecisionTreeClassifier(random_state=self.seed), cv=ShuffleSplit(n_splits=5,random_state=self.seed)) 
        scores_RF={}
        #scores_DT= answer_D["scores"]
        scores_RF["mean_fit_time"] = np.mean(scores_rf['fit_time'])
        scores_RF["std_fit_time"] = np.std(scores_rf['fit_time'])
        scores_RF["mean_accuracy"] = np.mean(scores_rf['test_score'])    
        scores_RF["std_accuracy"] = np.std(scores_rf['test_score'])  

        """scores_RF["model_highest_accuracy"] = np.max(scores_rf['test_score'])
        scores_RF["model_lowest_variance"] = np.min(np.var(scores_rf['test_score']))
        scores_RF["model_fastest"] = np.min(scores_rf['fit_time'])"""
        
        """scores_DT["mean_fit_time"] = np.mean(scores_dt['fit_time'])
        scores_DT["std_fit_time"] = np.std(scores_dt['fit_time'])
        scores_DT["mean_accuracy"] = np.mean(scores_dt['test_score'])    
        scores_DT["std_accuracy"] = np.std(scores_dt['test_score'])  
        scores_DT["model_highest_accuracy"] = np.max(scores_dt['test_score'])
        scores_DT["model_lowest_variance"] = np.min(np.var(scores_dt['test_score']))
        scores_DT["model_fastest"] = np.min(scores_dt['fit_time'])  """

        answer["clf_RF"] =  RandomForestClassifier(random_state=self.seed)
        answer["clf_DT"] = answer_D["clf"]
        answer["cv"] = ShuffleSplit(n_splits=5,random_state=self.seed)
        answer["scores_RF"] = scores_RF
        answer["scores_DT"] = answer_D["scores"]
        answer["model_highest_accuracy"] = "Random Forest" if scores_RF["mean_accuracy"] > answer_D["scores"]["mean_accuracy"] else "Decision Tree"
        answer["model_lowest_variance"] = min((scores_RF["std_accuracy"]**2),(answer_D["scores"]["std_accuracy"]**2)) ##String comes here
        answer["model_fastest"] = min(scores_RF["mean_fit_time"],answer_D["scores"]["mean_fit_time"]) ## String comes here

        

        # Enter your code, construct the `answer` dictionary, and return it.

        """
         Answer is a dictionary with the following keys: 
            "clf_RF",  # Random Forest class instance
            "clf_DT",  # Decision Tree class instance
            "cv",  # Cross validator class instance
            "scores_RF",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "scores_DT",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "model_highest_accuracy" (string)
            "model_lowest_variance" (float)
            "model_fastest" (float)
        """

        return answer

    # ----------------------------------------------------------------------
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """

    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - clf: The classifier instance to use for classification.
        - cv: The cross validator instance to use for cross validation.
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """
        # X, y, Xtest, ytest = u.prepare_data()
        # X, y = u.filter_out_7_9s(X, y)
        # Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        answer_F = self.partF(X=X,y=y)
        # Initialize the Random Forest Classifier 
        rf_clf = answer_F["clf_RF"]
        rf_clf.fit(X,y)
        y_train_pred = rf_clf.predict(X)
        y_test_pred = rf_clf.predict(Xtest)
        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=ShuffleSplit(n_splits=5,random_state=self.seed), scoring='accuracy', n_jobs=-1)
        # Perform grid search
        grid_search.fit(X, y)
        best_params= grid_search.best_estimator_
        #print("best_PARAMS:",best_params)
        accuracy = best_params.score(Xtest,ytest)
        mean_test_scores = grid_search.cv_results_['mean_test_score']
        # Calculate the mean accuracy
        mean_accuracy = mean_test_scores.mean()
        best_rf_clf = best_params
        best_rf_clf.fit(X,y) ##Doubtful
        y_best_train_pred = best_rf_clf.predict(X)
        y_best_test_pred = best_rf_clf.predict(Xtest)

        # Compute the confusion matrix
        confusion_matrix_train_orig = confusion_matrix(y, y_train_pred)
        confusion_matrix_test_orig = confusion_matrix(ytest,y_test_pred)
        confusion_matrix_train_best = confusion_matrix(y,y_best_train_pred)
        confusion_matrix_test_best = confusion_matrix(ytest,y_best_test_pred)

        
        


        # refit=True: fit with the best parameters when complete
        # A test should look at best_index_, best_score_ and best_params_
        """
        List of parameters you are allowed to vary. Choose among them.
         1) criterion,
         2) max_depth,
         3) min_samples_split, 
         4) min_samples_leaf,
         5) max_features 
         5) n_estimators
        """

        
        ### accuracy calculated out of confusion matrix 
        def calculate_accuracy(confusion_matrix):
            """
            Calculate accuracy from a confusion matrix.

            Parameters:
                confusion_matrix: 2D numpy array representing the confusion matrix.

            Returns:
                Accuracy computed from the confusion matrix.
            """
            # Calculate accuracy from confusion matrix
            TP = confusion_matrix[1, 1]  # True Positives
            TN = confusion_matrix[0, 0]  # True Negatives
            total_samples = confusion_matrix.sum()  # Total Samples

            accuracy = (TP + TN) / total_samples
            return accuracy

        def compute_accuracies(confusion_matrix_train_orig, confusion_matrix_test_orig, confusion_matrix_train_best, confusion_matrix_test_best):
            """
            Compute accuracies for each confusion matrix.

            Parameters:
                confusion_matrix_train_orig: Confusion matrix for training data with original estimator.
                confusion_matrix_test_orig: Confusion matrix for testing data with original estimator.
                confusion_matrix_train_best: Confusion matrix for training data with best estimator.
                confusion_matrix_test_best: Confusion matrix for testing data with best estimator.

            Returns:
                A dictionary containing accuracies for each confusion matrix.
            """
            accuracies = {}

            # Calculate accuracy for each confusion matrix
            accuracies["accuracy_orig_full_training"] = calculate_accuracy(confusion_matrix_train_orig)
            accuracies["accuracy_orig_full_testing"] = calculate_accuracy(confusion_matrix_test_orig)
            accuracies["accuracy_best_full_training"] = calculate_accuracy(confusion_matrix_train_best)
            accuracies["accuracy_best_full_testing"] = calculate_accuracy(confusion_matrix_test_best)

            return accuracies

        # Example usage:
        # Assuming you have four confusion matrices: confusion_matrix_train_orig, confusion_matrix_test_orig,
        # confusion_matrix_train_best, confusion_matrix_test_best
        accuracies = compute_accuracies(confusion_matrix_train_orig, confusion_matrix_test_orig,
                                        confusion_matrix_train_best, confusion_matrix_test_best)
        
        answer = {}
        answer["clf"] = answer_F["clf_RF"]
        answer["default_parameters"] = rf_clf.get_params()
        answer["best_estimator"] = best_params
        answer["grid_search"] = grid_search
        answer["mean_accuracy_cv"] = mean_accuracy
        answer["confusion_matrix_train_orig"] = confusion_matrix_train_orig
        answer["confusion_matrix_test_orig"] =  confusion_matrix_test_orig
        answer["confusion_matrix_train_best"] = confusion_matrix_train_best
        answer["confusion_matrix_test_best"] = confusion_matrix_test_best
        answer["accuracy_orig_full_training"] = accuracies["accuracy_orig_full_training"]
        answer["accuracy_orig_full_testing"] = accuracies["accuracy_orig_full_testing"]
        answer["accuracy_best_full_training"] = accuracies["accuracy_best_full_training"]
        answer["accuracy_best_full_testing"] = accuracies["accuracy_best_full_testing"]
        # Enter your code, construct the `answer` dictionary, and return it.

        """
           `answer`` is a dictionary with the following keys: 
            
            "clf", base estimator (classifier model) class instance
            "default_parameters",  dictionary with default parameters 
                                   of the base estimator
            "best_estimator",  classifier class instance with the best
                               parameters (read documentation)
            "grid_search",  class instance of GridSearchCV, 
                            used for hyperparameter search
            "mean_accuracy_cv",  mean accuracy score from cross 
                                 validation (which is used by GridSearchCV)
            "confusion_matrix_train_orig", confusion matrix of training 
                                           data with initial estimator 
                                (rows: true values, cols: predicted values)
            "confusion_matrix_train_best", confusion matrix of training data 
                                           with best estimator
            "confusion_matrix_test_orig", confusion matrix of test data
                                          with initial estimator
            "confusion_matrix_test_best", confusion matrix of test data
                                            with best estimator
            "accuracy_orig_full_training", accuracy computed from `confusion_matrix_train_orig'
            "accuracy_best_full_training"
            "accuracy_orig_full_testing"
            "accuracy_best_full_testing"
               
        """

        return answer
