import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from model_selection import get_best_models, get_test_data

x_test, y_test = get_test_data()
rfc, bt, lr, lda, knn = get_best_models().values()



# globals
C = 'darkorange'
FEATURE_NAMES = ['Male', 'Incumbent', 'Indigenous', 'Lawyer', 'Switcher', 'Multiple Candidacy',
                 'Governing Party', 'Liberal', 'Conservative', 'NDP', 'Green', 'Other Party', 'oc.Business',
                 'oc.Health', 'oc.Management', 'oc.MP', 'oc.Natural Science', 'oc. Natural Resources', 'oc.Entertainment',
                 "oc.Law, Education, Gov't", 'oc.Manufacturing', 'oc.Sales', 'oc.Trades']


fig1, axs = plt.subplots(1, 2, sharey=True, figsize=(10,5))
# # random forest
feature_importances = pd.Series(rfc.feature_importances_, index=FEATURE_NAMES)
# fig1 = plt.figure(1)
feature_importances.plot.barh(color=C, ax=axs[0])
axs[0].set_title('Random Forest')
fig1.supylabel('Feature')
fig1.supxlabel('Mean Decrease in Impurity')
# fig1.tight_layout()

# # boosting tree
feature_importances2 = pd.Series(bt.feature_importances_, index=FEATURE_NAMES)
# fig2 = plt.figure(2)
feature_importances2.plot.barh(color=C, ax=axs[1])
axs[1].set_title('Boosting Trees')
# axs[1].set_ylabel('Feature')
# axs[1].set_xlabel('Mean Decrease in Impurity')
fig1.tight_layout()


fig2, axs = plt.subplots(1, 2, sharey=True, figsize=(10,5))
# # logistic regression
coefs = pd.Series(lr.coef_[0], index=FEATURE_NAMES)
# fig3 = plt.figure(3)
coefs.plot.barh(color=C, ax=axs[0])
axs[0].set_title('Logistic Regression')
fig2.supylabel('Feature')
fig2.supxlabel('Coefficient Value')
fig2.tight_layout()

# lda
coefs2 = pd.Series(lda.coef_[0], index=FEATURE_NAMES)
# fig4 = plt.figure(4)
coefs2.plot.barh(color=C, ax=axs[1])
axs[1].set_title('Linear Discriminant Analysis')
# plt.ylabel('Feature')
# plt.xlabel('Coefficient Value')
# fig4.tight_layout()

# knn



if __name__ == '__main__':
    plt.show()
