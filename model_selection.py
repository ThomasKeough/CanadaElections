import pandas as pd
from clean import clean_dataset
import json
import numpy as np

# model imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# other methods
from sklearn.model_selection import train_test_split, GridSearchCV

# globals
PATH = 'data/federal-candidates-2021-10-20.csv'
RAND_STATE = 72

# build df
df = clean_dataset(PATH)


# df = df.dropna(subset=['birth_year']) # drops only observations with multiple candidacy

trudeau = df[df['candidate_name'] == "Trudeau, Justin"].iloc[1]
trudeau = trudeau.drop('id')
trudeau = trudeau.drop('candidate_name')
trudeau = trudeau.drop('elected')
trudeau = np.array(trudeau).reshape(1, -1)


# set up x, y
x = df.drop(columns=['id', 'candidate_name', 'elected'])
y = df['elected']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RAND_STATE)

# ====== 10-fold CV per model =======
models = [RandomForestClassifier(), GradientBoostingClassifier(),
           LogisticRegression(), LinearDiscriminantAnalysis(),
           KNeighborsClassifier()]

# random forest classifier
rfc_pars = {'max_depth': [3, 6, 9, 12, 5], 'random_state': [RAND_STATE]}

# boosting tree
bt_pars = {'learning_rate': (0.05, 0.1, 0.15, 0.20, 0.25, 0.30), 
           'max_depth': [3, 6, 9, 12, 15],
           'random_state': [RAND_STATE]}

# logistic regression
lr_pars = {'penalty': ['l2'], 'C': [1, 5, 10], 'max_iter': [400]} # best was C = 1
# lr_pars = {'penalty': ['l2'], 'C': [1, 2, 3, 4], 'max_iter': [200]}

# linear discriminant analysis
lda_pars = {'solver': ['lsqr']}

# k-nearest neighbours
knn_pars = {'n_neighbors': [5, 10, 15, 20]} # best was 5

pars = [rfc_pars, bt_pars, lr_pars, lda_pars, knn_pars]

# build gridsearchcv object for each classifer; 10 fold cross validation
gscv = [GridSearchCV(model, par, cv=10).fit(x_train, y_train) for model, par in zip(models, pars)]
best_params = [cv.best_params_ for cv in gscv]


# uncomment to run best models
# best_models = {'random forest': RandomForestClassifier(max_depth=9, random_state=72).fit(x_train, y_train),
#                'boosting tree': GradientBoostingClassifier(learning_rate=0.15, max_depth=6, random_state=72).fit(x_train, y_train),
#                'logistic regression': LogisticRegression(C=1, max_iter=400, penalty='l2').fit(x_train, y_train),
#                'linear discriminant analysis': LinearDiscriminantAnalysis(solver='lsqr').fit(x_train, y_train),
#                'k nearest neighbors': KNeighborsClassifier(n_neighbors=15).fit(x_train, y_train)}

# performance = {name: model.score(x_test, y_test) for name, model in best_models.items()}

def get_best_models():
    return best_models
    
def get_test_data():
    """returns (x_test, y_test).
    """
    return (x_test, y_test)


if __name__ == '__main__':
    # print(best_params)
    # with open('scores.json', 'w') as f:
    #     f.write(json.dumps(performance, indent=6))
    print([model.predict(trudeau) for name, model in best_models.items()])
