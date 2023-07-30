from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, f_classif, SelectKBest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from mlxtend.classifier import EnsembleVoteClassifier
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

FILEPATH = './genres_v2.csv'
df = pd.read_csv(FILEPATH)

# Join columns title and song_name
titles = df[["song_name", "title"]]
titles = titles["song_name"].combine_first(titles['title'])
df['song_name'] = titles

# Create a new col with a num counterpart of genre
num_genres = range(15)
GENRES = list(df.copy().groupby('genre').count().index)  # number to genre
genre_to_num = dict(zip(GENRES, num_genres))
tmp = df['genre'].copy(deep=True).replace(genre_to_num)
df['genre_num'] = tmp

# Fill n/a values of song_name into unnamed
df["song_name"] = df["song_name"].fillna("unnamed")

# Remove irrelevant columns from the dataset (remove metadata)
to_remove = ["type", "id", "uri", "track_href", "analysis_url", "title", "Unnamed: 0"]
for rm in to_remove:
    del df[rm]

print(df)
print("\nDtypes\n", df.dtypes)
print("\nSUMMARY\n", df.describe(include='number').T, end='\n\n')
print("genre_num to genre mapping")
for i, genre in enumerate(GENRES):
    print(i, genre)
print()

# exit()
# ** SPLIT INTO X AND Y **
# Split data into X and y
X = df.copy()
del X['genre_num']
del X['genre']

y = df['genre_num']

X = X.select_dtypes(include='number')
print(f"Features ({len(X.columns)}):", list(X.columns))
print("Target column:", y.name)

# ** SMOTE **
X, y = SMOTE().fit_resample(X, y)

# ** SCALING **
X_cpy = X.copy()
sc_features = X_cpy[
    ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
     'tempo', 'duration_ms']]
unscaled_features = X_cpy[['key', 'mode', 'time_signature']]
sc_x = StandardScaler().fit(sc_features.values)
scaled = sc_x.transform(sc_features.values)
sc_x_features = pd.DataFrame(scaled, index=X.index, columns=sc_features.columns)
X = pd.concat([sc_x_features, unscaled_features], axis='columns')
print("\nUnscaled feature columns:", list(unscaled_features.columns.values))
print("Scaled feature columns:", list(sc_features.columns.values))
print("\nScaled X:", X)
print("\nSUMMARY of transformed features:\n", X.describe(include='number').T, end='\n\n')


## ** FEATURE SELECT **
def select_features(selector, selector_name, X, y):
  selector.fit(X, y)
  selected = list(selector.get_feature_names_out())
  selected_features = {}
  selected_features[selector_name] = selected_features
  return selected


rfe = RFE(RandomForestClassifier(n_estimators=100), step=5, n_features_to_select=10)
print("RFE:", select_features(rfe, 'rfe', X, y))

ffs = SelectKBest(score_func=f_classif, k=5)
print("FFS", select_features(ffs, 'ffs', X, y))

def showFeatureImportances(clf_model, X, y):
    print(f"** {clf_model.__class__.__name__} feature importance **")
    clf_model.fit(X, y)
    importances = list(clf_model.feature_importances_)

    dfImportance = pd.DataFrame()
    selected = []
    for i in range(0, len(importances)):
        dfImportance = dfImportance.append({"importance": importances[i], "feature": X.columns[i]}, ignore_index=True)
        if importances[i] > 0.05:
            selected.append(X.columns[i])
    dfImportance = dfImportance.sort_values(by=['importance'], ascending=False)
    print(dfImportance)
    print("SELECTED by feature importance > 0.05:", selected)


showFeatureImportances(RandomForestClassifier(n_estimators=200), X, y)

# Declare the best features
best_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence',
                 'liveness', 'tempo', 'duration_ms']
X = X[best_features]
print("Selected Features:", list(X.columns))


## ** MODEL TRAINING (Xfold validation) **
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)

    evals = {}

    evals['cm'] = confusion_matrix(y_test, y_pred)
    evals["accuracy"] = accuracy_score(y_test, y_pred)
    evals["precision"] = precision_score(y_test, y_pred, average='macro')
    evals["recall"] = recall_score(y_test, y_pred, average='macro')
    evals["f1"] = f1_score(y_test, y_pred, average='macro')
    evals["AUC"] = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')

    return evals, y_pred


k = 5
kfold = KFold(k, shuffle=True)
results = pd.DataFrame(columns=('cm', 'accuracy', 'precision', 'recall', 'f1', 'AUC'))

clfs = [
    LogisticRegression(),
    RandomForestClassifier(n_estimators=200),
    BaggingClassifier(KNeighborsClassifier(), n_estimators=10),
    EnsembleVoteClassifier(clfs=[XGBClassifier(), RandomForestClassifier(n_estimators=100), KNeighborsClassifier()],
                           voting='hard')
]
for clf in clfs:
    i = 0
    print(f"** Training {clf.__class__.__name__} **")
    for train, test in kfold.split(X, y):
        print(f"\nTrain size: {len(train)}", f"Test size: {len(test)}")
        train_x, test_x = X.iloc[train], X.iloc[test]
        train_y, test_y = y.iloc[train], y.iloc[test]

        # Create model
        model = clf.fit(train_x, train_y)
        print(f"Model {i} fitting done")

        # Evaluate metrics
        evals, preds = evaluate(model, test_x, test_y)

        results.loc[f"Model {i}"] = evals
        print(f"Model {i} eval done")
        i += 1

    # Show metrics
    print(results)
    print()
    averages = {}
    for col in results.columns:
        if col not in ['cm']:
            key = f"Average {col}"
            averages[key] = results[col].mean()
            print(key + ":", averages[key])
            print(f"Std dev {col}:", results[col].std())


# ** Stacked Model **
print("** STACKED MODEL **")
def fitBaseModels(X_train, y_train, X_test, models):
    dfPredictions = pd.DataFrame()
    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        colName = str(i)
        # Add base model predictions to column of data frame.
        dfPredictions[colName] = predictions
    return dfPredictions, models


def fitStackedModel(X, y):
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    return model


def evaluate_print(y_true, y_pred):
    evals = {}

    # evals['cm'] = confusion_matrix(y_true, y_pred)
    evals["accuracy"] = accuracy_score(y_true, y_pred)
    evals["precision"] = precision_score(y_true, y_pred, average='macro')
    evals["recall"] = recall_score(y_true, y_pred, average='macro')
    evals["f1"] = f1_score(y_true, y_pred, average='macro')

    for x, y in evals.items():
        print(f"\t{x}", y)
    return evals


# Split data into train, test and validation sets.
k = 5
kfold = KFold(k, shuffle=True)
j = 0
results = pd.DataFrame(columns=('cm', 'accuracy', 'precision', 'recall', 'f1', 'AUC'))
for train, test in kfold.split(X, y):
    X_train, X_temp = X.iloc[train], X.iloc[test]
    y_train, y_temp = y.iloc[train], y.iloc[test]

    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

    # Fit base and stacked models.
    model_stack = [
        XGBClassifier(),
        ExtraTreesClassifier(n_estimators=100),
        RandomForestClassifier(n_estimators=100),
        KNeighborsClassifier()
    ]
    dfPredictions, models = fitBaseModels(X_train, y_train, X_val, model_stack)
    stackedModel = fitStackedModel(dfPredictions, y_val)

    # Evaluate base models with validation data.
    print(f"\n** Evaluate Base Models {j} **")
    dfValidationPredictions = pd.DataFrame()
    for i in range(0, len(models)):
        predictions = models[i].predict(X_test)
        colName = str(i)
        dfValidationPredictions[colName] = predictions
        print(models[i].__class__.__name__)
        base_evals, base_pred = evaluate(models[i], X_test, y_test)
        for eval_key, eval_val in base_evals.items():
            if eval_key not in ['cm']:
                print(f"\t{eval_key}", eval_val)
        print()

    # Evaluate stacked model with validation data.
    stackedPredictions = stackedModel.predict(dfValidationPredictions)
    print(f"\n** Evaluate Stacked Model {j} **")
    evals, _ = evaluate(stackedModel, dfValidationPredictions, y_test)
    results.loc[f"Model {j}"] = evals
    j += 1
# Show metrics
print(results)
print()
averages = {}
for col in results.columns:
    if col not in ['cm']:
        key = f"Average {col}"
        averages[key] = results[col].mean()
        print(key + ":", averages[key])
        print(f"Std dev {col}:", results[col].std())

