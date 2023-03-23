# AI-fairness


This is where I will be committing some implementations for Data/Model Fairness metrics.
These metrics will be accompanied by definitions and other relevant information as they are developed.



****Current Implementations****

** Data Level Metrics **

Disparate Impact: 
The Disparate Impact value is the ratio of the probability of a sensitive attribute value having a positive
outcome and the probability of the reference attribute value having a positive outcome. The metric assumes these
probabilities are equal if the labels are unbiased with respect to the sensitive attribute.
In general, a DI value less than 0.8 or greater than 1.25 indicates bias with respect to
the reference group.
With the following mathematical representation:
DI = \frac{P(Y = +|*S*<sub>*i*</sub>)}{P(Y = +|*S*<sub>*r*</sub>)}
Note: This metric could be modified for prediction/model bias evaluation.

Statistical Parity Difference: 
The Statistical Parityy Difference (SPD) value is the difference between the probability
of the sensitive attribute value having a positive outcome and the probability of the 
reference attribute value having a positive outcome. The metric assumes that the two
probabilities (or statistical parities) are equal if the labels are unbiased,
and thus an expected value for "true fairness" is 0.

** Model Level Metrics **

Proportional Parity:
Proportional parity is a calculation based on the proportion of all positively
classified members in all subgroups of the data, using the following formula:
(TP + FP) / (TP + FP + TN + FN). 

The output will return a value of 1 for the reference group, and the other groups will be assigned 
values corresponding to whether their positively predicted observations are greater or less than the
reference groups. Lower proportions will be reflected in numbers lower than 1, and higher proportions
will be reflected in numbers greater than 1.

Equal Opportunity Difference:
The Equal Opportunity Difference (EOD) value is the difference of the True Positive Rate (TPR) between a 
given sensitive attribute *S*<sub>*i*</sub> for a group *S* and the reference attribute for that group *S*<sub>*r*</sub>, 
represented as such:
EOD = TPR(*S*<sub>*i*</sub>) - TPR(*S*<sub>*r*</sub>)

Average Absolute odds Difference:

