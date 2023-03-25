# Metric: Average Absolute Odds Difference


from sklearn.metrics import confusion_matrix


def average_absolute_odds_difference(dataset, true_labels, predictions, attribute, reference=None, target_value=None):
    """
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
    This function computes the Average Absolute Odds Difference, which measures bias
    through the false positive rate and true positive rate. The formula for this metric is:

    Average Absolute Odds Difference (AAOD)
     = 1/2 * [|FPR(minority) - FPR(reference/majority)| + |TPR(minority) - TPR(reference/majority)|]
    for a group of a sensitive attribute.
    Where FPR is the False Positive Rate and TPR is the True Positive Rate.
    FPR = False Positive / (False Positive + True Negative)
    TPR = True Positive / (True Positive + False Negative)
    The output will return the AAOD for each class in the chosen sensitive attribute,
    if a reference class is not given, the function will default to the class that is
    the majority in the dataset.

    Output: An AAOD value closer to zero is an indicator of the model being more fair,
    with an AAOD of zero is true fairness.

    """

    # Get most common as reference if left null.
    if reference is None:
        reference = max(set(dataset[attribute]))
    # Set positive outcome value as 1 (Binary Positive) if left blank.
    if target_value is None:
        target_value = 1
    dataset['true_labels'] = true_labels
    dataset['predictions'] = predictions
    attribute_components = dataset[attribute].unique()
    ref_dataset = dataset.loc[(dataset[attribute] == reference)]
    true_neg_ref, false_pos_ref, false_neg_ref, true_pos_ref = confusion_matrix(ref_dataset['true_labels'],
                                                                                ref_dataset['predictions']).ravel()
    ref_tpr = true_pos_ref / (true_pos_ref + false_neg_ref)
    ref_fpr = false_pos_ref / (false_pos_ref + true_neg_ref)
    for comp in attribute_components:

        if comp == reference:
            print(f"{comp} is the reference class.")
        else:
            comp_dataset = dataset.loc[(dataset[attribute] == comp)]
            tn1, fp1, fn1, tp1 = confusion_matrix(comp_dataset['true_labels'], comp_dataset['predictions']).ravel()
            comp_tpr = tp1 / (tp1 + fn1)
            comp_fpr = fp1 / (fp1 + tn1)
            tpr_diff = abs(comp_tpr - ref_tpr)
            fpr_diff = abs(comp_fpr - ref_fpr)

            avg_abs_odds_diff = 1/2 * (tpr_diff + fpr_diff)

            print(f"The Equal Opportunity Difference for the protected attribute {comp} is {avg_abs_odds_diff}")
