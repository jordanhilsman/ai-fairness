# AI-fairness


This is where I will be committing some implementations for Data/Model Fairness metrics.
These metrics will be accompanied by definitions and other relevant information as they are developed.



****Current Implementations****

** Data Level Metrics **

Disparate Impact:

Statistical Parity Difference:

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

Average Absolute odds Difference:

