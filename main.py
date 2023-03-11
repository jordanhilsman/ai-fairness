# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from proportional_parity import *
from statistical_parity_diff import *
from disparate_impact import *
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'target']

# load dataset
df = pd.read_csv(r'C:\Users\Jordan\Downloads\adult.csv', names=names, na_values=' ?')
df = df.dropna()

# targets
target = [1 if tar == ' >50K' else 0 for tar in df['target']]

# features, dummy variables for logistic regression
X = df[['age', 'education-num', 'hours-per-week']]
X['marital-status'] = [1 if x == ' Married-civ-spouse' else 0 for x in df['marital-status']]
X['native-country'] = [1 if x == ' United States' else 0 for x in df['native-country']]

occ_groups = {
        ' Priv-house-serv': 0, ' Other-service': 1, ' Handlers-cleaners': 2,
        ' Farming-fishing': 3, ' Machine-op-inspct': 4, ' Adm-clerical': 5,
        ' Transport-moving': 6, ' Craft-repair': 7, ' Sales': 8,
        ' Armed-Forces': 9, ' Tech-support': 10, ' Protective-serv': 11,
        ' Prof-specialty': 12, ' Exec-managerial': 13}

race = {
        ' White': 0, ' Black': 1, ' Asian-Pac-Islander': 2, ' Amer-Indian-Eskimo': 3, ' Other': 4
}



X['occupation'] = [occ_groups[x] for x in df['occupation']]
X['race'] = [race[x] for x in df['race']]

# created dataframe without the target column for ease in Proportional Parity library.
no_target = X.loc[:, X.columns != "target"]
X['target'] = target

# Perform train/test split for predictions and assessment.
X_train, X_test, y_train, y_test = train_test_split(no_target, target, test_size=0.2, random_state=3)
model = LogisticRegression()

model.fit(X_train, y_train)
# Predict outcomes with the model
y_preds = model.predict(X_test)
# Execute proportional parity test - In this case, we are examining on the feature 'race', with ' White'
# set as the baseline.
#prop_parity(X_test, y_test, race, ' Black', y_preds, 'race')
#prop_parity(X_test, y_test, occ_groups, ' Adm-clerical', y_preds, 'occupation')

#Statistical_Parity_Diff(df, 'race', 'target')

dff = df
dff['target'] = target
#print(dff['target'].value_counts()) #30162
#print(dff['race'].value_counts()) # White - 25933, Black - 2817, API - 895, AIE-286, Other-231
#print(dff['race'].value_counts().sum()) #30162
df2 = df.loc[(dff['race'] == ' Black') & (dff['target'] == 1)]
#print(df2)
#print(len(dff.loc[(dff['race'] == ' Black') & (dff['target'] == 1)]))
#print(list(df['race'].unique()))
#30162

#print(len(dff.loc[(dff['race'] == ' White') & (dff['target'] == 1)]))
#print(len(dff.loc[(dff['race'] == ' Black') & (dff['target'] == 1)]))
#print(len(dff.loc[(dff['race'] == ' Asian-Pac-Islander') & (dff['target'] == 1)]))
#print(len(dff.loc[(dff['race'] == ' Amer-Indian-Eskimo') & (dff['target'] == 1)]))
#print(len(dff.loc[(dff['race'] == ' Other') & (dff['target'] == 1)]))

#print(len(dff.loc[(dff['race'] == ' White')]))
#print(len(dff.loc[(dff['race'] == ' Black')]))
#print(len(dff.loc[(dff['race'] == ' Asian-Pac-Islander')]))
#print(len(dff.loc[(dff['race'] == ' Amer-Indian-Eskimo')]))
#print(len(dff.loc[(dff['race'] == ' Other')]))




Statistical_Parity_Diff(df, 'race', 'target')
Disparate_Impact(df, 'race', 'target', ' Black')
