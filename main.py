# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from proportional_parity import *
from statistical_parity_diff import *
from disparate_impact import *
from equal_opportunity_difference import *
from average_absolute_odds_difference import *
import numpy as np

names = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "target",
]

# load dataset
df = pd.read_csv(
    r"~/Documents/pittnail/ai-fairness-main/adult.csv", names=names, na_values=" ?"
)
df = df.dropna()
df["race"] = df["race"].str.strip()
df["marital-status"] = df["marital-status"].str.strip()
df["native-country"] = df["native-country"].str.strip()
df["target"] = df["target"].str.strip()
df["occupation"] = df["occupation"].str.strip()

# targets
target = [1 if tar == " >50K" else 0 for tar in df["target"]]


# features, dummy variables for logistic regression
df["marital-status"] = np.where(df["marital-status"] == "Married-civ-spouse", 1, 0)
df["native-country"] = np.where(df["native-country"] == "United States", 1, 0)
df["target"] = np.where(df["target"] == ">50K", 1, 0)

# perform data-bias early.
statistical_parity_diff(df, "race", "target")
disparate_impact(df, "race", "target")
occ_groups = {
    "Priv-house-serv": 0,
    "Other-service": 1,
    "Handlers-cleaners": 2,
    "Farming-fishing": 3,
    "Machine-op-inspct": 4,
    "Adm-clerical": 5,
    "Transport-moving": 6,
    "Craft-repair": 7,
    "Sales": 8,
    "Armed-Forces": 9,
    "Tech-support": 10,
    "Protective-serv": 11,
    "Prof-specialty": 12,
    "Exec-managerial": 13,
}

race = {
    "White": 0,
    "Black": 1,
    "Asian-Pac-Islander": 2,
    "Amer-Indian-Eskimo": 3,
    "Other": 4,
}

df["occupation"] = [occ_groups[x] for x in df["occupation"]]
df["race"] = [race[x] for x in df["race"]]
X = df[
    [
        "age",
        "education-num",
        "hours-per-week",
        "marital-status",
        "native-country",
        "occupation",
        "race",
        "target",
    ]
]


# created dataframe without the target column for ease in Proportional Parity library.
no_target = X.loc[:, X.columns != "target"]

# Perform train/test split for predictions and assessment.
X_train, X_test, y_train, y_test = train_test_split(
    no_target, X["target"], test_size=0.2, random_state=3
)
model = LogisticRegression()

model.fit(X_train, y_train)
# Predict outcomes with the model
y_preds = model.predict(X_test)

dff = df
dff["target"] = target
df2 = df.loc[(dff["race"] == "Black") & (dff["target"] == 1)]


equal_opportunity_difference(X_test, y_test, y_preds, "race", 0, 1)
prop_parity(X_test, y_test, race, "White", y_preds, "race")
average_absolute_odds_difference(X_test, y_test, y_preds, "race")
