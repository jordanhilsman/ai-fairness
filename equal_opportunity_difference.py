# Metric: Equal Opportunity Difference

'''
This function computes the Equal Opportunity Difference metric which can be expressed mathematically as:
EOD = P(Y^ = 1 | Y = 1, A = minority) - P(Y^ = 1 | Y =1,  A = reference), where Y^ are the model
prediction labels, Y are the true labels, and A is the sensitive group in question (i.e. race, sex, etc.)
To represent this difference, we will use the following formula:
EOD = ((TRUE POSITIVE RATE (MINORITY CLASS) - TRUE POSITIVE RATE (MAJORITY OR REFERENCE CLASS))

Equal Opportunity Difference (EOD) measures the deviation form the equality of opportunity,
meaning that the same proportion of each population receives the favorable outcome.
This measure must be equal to 0 for true fairness.

The output will return the EOD for each class in the chosen sensitive attribute,
if a reference class is not given, the function will default to the class that is
the majority in the dataset.

Parameters:
dataset: The test dataset that the model is tested against.
true_labels: The target labels that were used in testing the model.
predictions: The series of predictions made by the model on the testing dataset.
attribute: column name for which perform statistical parity difference is being measured across constituent classes.
target: column where true labels are given.
reference (optional): Reference class for the EOD function, if left null will default to most frequent member class
of the given attribute.
target value (optional): Input value that corresponds to a positive outcome in the target column, if left blank,
will default to 1.

Output: a negative EOD is a bias against an attribute class, while a positive SPD is a bias *for* that attribute class.
'''
from sklearn.metrics import confusion_matrix

def equal_opportunity_difference(dataset, true_labels, predictions, attribute, reference=None, target_value=None):

    # Get most common as reference if left null.
    if reference is None:
        reference = max(set(dataset[attribute]))
    else:
        pass
    # Set positive outcome value as 1 (Binary Positive) if left blank.
    if target_value is None:
        target_value = 1
    else:
        pass
    dataset['true_labels'] = true_labels
    dataset['predictions'] = predictions
    attribute_components = list(dataset[attribute].unique())
    ref_dataset = dataset.loc[(dataset[attribute] == reference)]
    for comp in attribute_components:

        if comp == reference:
            print(comp, " is the reference class.")
        else:
            comp_dataset = dataset.loc[(dataset[attribute] == comp)]
            tn1, fp1, fn1, tp1 = confusion_matrix(comp_dataset['true_labels'], comp_dataset['predictions']).ravel()
            true_pos_rate_comp = (tp1 / (tp1 + fn1))
            tn2, fp2, fn2, tp2 = confusion_matrix(ref_dataset['true_labels'], ref_dataset['predictions']).ravel()
            true_pos_rate_ref = (tp2 / (tp2 + fn2))
            eod = true_pos_rate_comp - true_pos_rate_ref
            print("The Equal Opportunity Difference for the protected attribute ", comp, " is ", eod)
