import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from pylab import rcParams
import seaborn as sb
import scipy
from scipy.stats.stats import pearsonr
# %matplotlib inline
rcParams['figure.figsize'] = 5,4
sb.set_style('whitegrid')

data = pd.read_csv("klasa.csv")
data.shape
data.head()

data.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32' ]

X = data[ ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32' ]]

corr = X.corr()
abs(corr)


# X bierze wszystko oprocz klasy
X = data.drop(['1', '5', '10', '12', '15', '16', '17', '18', '19',
'21', '22', '23', '24', '25', '27', '28', '29', '30', '31', '32' ], axis=1)
# w Y sa tylko kklasy
y = data['32']

from sklearn.feature_selection import SelectKBest, chi2

def get_k_best_features(X, y, k):
    X_dataframe = pd.DataFrame(X)
    selector = SelectKBest(chi2, k)
    selector.fit(X_dataframe, y)
    chi_support = selector.get_support()
    selected_features = X_dataframe.loc[:,chi_support]
    return selected_features
for x in range(0, 5):

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    from sklearn.svm import SVC
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix
    print("Confussion Matrix: \n")
    print(confusion_matrix(y_test,y_pred))
    print("\n")
    print("classification report: \n")
    print(classification_report(y_test,y_pred))
    print("\n")
    print("\n")
