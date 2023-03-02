# Metric: Proportional Parity

'''
This function computes the Proportional parity metric with the following formula:
(TP + FP) / (TP + FP + TN + FN)

Proportional parity is calculated based on the proportion of all positively
classified members in all subgroups of the data.

The output will return the reference group with a 1, and the other groups will be
assigned values corresponding to whether their positively predicted observations
are greater or less than the reference groups. Lower proportions will be reflected
in numbers lower than 1.

Parameters:
data: input dataframe with necessary columns
outcome: column name that has outcome variable (target column)
group: Column name indicating group to evaluation proportional parity on.
base: Reference group for proportional parity.
predictions: predicted outcome vector
'''

import pandas as pd
from sklearn.metrics import accuracy_score


def prop_parity(data, outcome, group, base, predictions):

    # first check if data is a dataframe object
    if isinstance(data, pd.DataFrame):
        pass
    else:
        data = data.to_frame()
    accuracy_array = []
    # check lengths
    if len(outcome) != len(predictions):
        print('Prediction vector and outcome vector must be of the same length!')
    else:
        pass
    group_break = group.unique()
    dictionary = dict(list(enumerate(group_break)))
    dictionary = {v: k for k, v in dictionary.items()}
    data['y_test'] = outcome
    data['y_preds'] = predictions
    for x in dictionary.values():
        final_df = data[data['race'] == x]
        race = group_break[x]
        acc = accuracy_score(final_df['y_test'], final_df['y_preds'])
        print('The accuracy for the ' + race + " subset is ", acc)
        accuracy_array.append(
            {
                'Group': group_break[x],
                'Accuracy': accuracy_score(final_df['y_test'], final_df['y_preds'])
            }
        )
    accuracy_array = pd.DataFrame(accuracy_array)
    baseline = base
    baseline_accuracy = accuracy_array[accuracy_array['Group']==(baseline)]['Accuracy'].item()
    accuracy_array['Proportional Parity'] = (accuracy_array['Accuracy'] / baseline_accuracy)
    print(accuracy_array)
