

def statistical_parity_diff(dataset, target, attribute, reference=None, target_value=None):
    """
    Metric: Statistical Parity Difference
    Parameters:
    data: Dataset being evaluated for fairness.
    attribute: column name for which perform statistical parity difference is being measured across constituent classes.
    target: column where true labels are given.
    reference (optional): Reference class for the SPD function, if left null will default to most frequent member class
    of the given attribute.
    target value (optional): Input value that corresponds to a positive outcome in the target column, if left blank,
    will default to 1.
    This function computes the Statistical Parity Difference metric which can be expressed mathematically as:
    SPD = P(Y = 1 | A = minority) - P(Y = 1 | A = reference), where Y are the positive outcomes,
    and A is the sensitive group in question (i.e. race, sex, etc.)
    To represent this difference, we will use the following formula:
    SPD = ((Number of Positive Instances (minority class) / (Number of Positive Instances (reference class)) -
        ((Number of Instances (minority class) / Number of Instances (reference class))

    Statistical parity difference (SPD) is a metric designed to ascertain the difference
    at which the reference and protected classes of the attribute receive a favorable outcome in a dataset or model.
    For true fairness, the measure must be equal to 0.

    The output will return the SPD for each class in the chosen sensitive attribute,
    if a reference class is not given, the function will default to the class that is
    the majority in the dataset.

    Output: a positive SPD is a bias against an attribute class, while a negative SPD is a bias *for* that attribute
     class.
    """
    # Get most common as reference if left null.
    if reference is None:
        reference = max(set(dataset[attribute]))
    # Set positive outcome value as 1 (Binary Positive) if left blank.
    if target_value is None:
        target_value = 1

    attribute_components = list(dataset[attribute].unique())
    for comp in attribute_components:
        if comp == reference:
            print(comp + " is the majority class.")
        else:
            pos_prot = len(dataset.loc[(dataset[attribute] == comp) & (dataset[target] == target_value)])
            pos_ref = len(dataset.loc[(dataset[attribute] == reference) & (dataset[target] == target_value)])
            pos_quotient = pos_prot / pos_ref

            inst_prot = len(dataset.loc[(dataset[attribute] == comp)])
            inst_ref = len(dataset.loc[(dataset[attribute] == reference)])
            inst_quotient = inst_prot/inst_ref

            spd = pos_quotient - inst_quotient

            print(f"The Statistical Parity Difference for the protected attribute {comp} is {spd}")
