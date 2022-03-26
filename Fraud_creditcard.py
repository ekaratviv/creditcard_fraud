# Basic libraries
import pandas as pd
import numpy as np

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)
% matplotlib
inline

# preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# Metrics Libraries
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Misc libraries
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns

df = pd.read_csv("/content/drive/MyDrive/credit_card_fraud.csv")
ax = sns.barplot(x="type", y="amount", hue="isFraud", data=df).set(title="Transaction type Vs Amount")
df["type"].unique()
df = df.dropna()
df1 = df.groupby('type')['isFraud'].sum()
df1
df1 = df1.to_frame().reset_index()
ax1 = sns.barplot(x="type", y="isFraud", data=df1).set(title='Number of Frauds per Transaction type')

# Reading the data
paysim = pd.read_csv('/content/drive/MyDrive/credit_card_fraud.csv')

# Looking at the data
paysim.head()

# Pivot table
paysim_pivot1 = pd.pivot_table(paysim, index=["type"],
                               values=['amount', 'isFraud', 'isFlaggedFraud'],
                               aggfunc=[np.sum, np.std], margins=True)

# Adding color gradient
cm = sns.light_palette("lightblue", as_cmap=True)
paysim_pivot1.style.background_gradient(cmap=cm)

# Pivot table
paysim_pivot2 = pd.pivot_table(paysim, index=["type"],
                               values=['amount', 'oldbalanceOrg', 'newbalanceOrig'],
                               aggfunc=[np.sum], margins=True)

# Adding style
paysim_pivot2.style \
    .format('{:.2f}') \
    .bar(align='mid', color=['lightblue']) \
    .set_properties(padding='5px', border='3px solid white', width='200px')

paysim_pivot3 = pd.pivot_table(paysim, index=["type"],
                               values=['amount', 'oldbalanceDest', 'newbalanceDest'],
                               aggfunc=[np.sum], margins=True)

# Adding style
paysim_pivot3.style \
    .format('{:.2f}') \
    .bar(align='mid', color=['darkblue']) \
    .set_properties(padding='5px', border='3px solid white', width='200px')

paysim = pd.read_csv('/content/drive/MyDrive/credit_card_fraud.csv', nrows=50000)

# Distribution of Amount
fig = px.box(paysim, y="amount")
fig.show()


def balance_diff(data):
    '''balance_diff checks whether the money debited from sender has exactly credited to the receiver
       then it creates a new column which indicates 1 when there is a deviation else 0'''
    # Sender's balance
    orig_change = data['newbalanceOrig'] - data['oldbalanceOrg']
    orig_change = orig_change.astype(int)
    for i in orig_change:
        if i < 0:
            data['orig_txn_diff'] = round(data['amount'] + orig_change, 2)
        else:
            data['orig_txn_diff'] = round(data['amount'] - orig_change, 2)
    data['orig_txn_diff'] = data['orig_txn_diff'].astype(int)
    data['orig_diff'] = [1 if n != 0 else 0 for n in data['orig_txn_diff']]

    # Receiver's balance
    dest_change = data['newbalanceDest'] - data['oldbalanceDest']
    dest_change = dest_change.astype(int)
    for i in dest_change:
        if i < 0:
            data['dest_txn_diff'] = round(data['amount'] + dest_change, 2)
        else:
            data['dest_txn_diff'] = round(data['amount'] - dest_change, 2)
    data['dest_txn_diff'] = data['dest_txn_diff'].astype(int)
    data['dest_diff'] = [1 if n != 0 else 0 for n in data['dest_txn_diff']]

    data.drop(['orig_txn_diff', 'dest_txn_diff'], axis=1, inplace=True)


# Surge indicator
def surge_indicator(data):
    '''Creates a new column which has 1 if the transaction amount is greater than the threshold
    else it will be 0'''
    data['surge'] = [1 if n > 450000 else 0 for n in data['amount']]


# Frequency indicator
def frequency_receiver(data):
    '''Creates a new column which has 1 if the receiver receives money from many individuals
    else it will be 0'''
    data['freq_Dest'] = data['nameDest'].map(data['nameDest'].value_counts())
    data['freq_dest'] = [1 if n > 20 else 0 for n in data['freq_Dest']]

    data.drop(['freq_Dest'], axis=1, inplace=True)


# Tracking the receiver as merchant or not
def merchant(data):
    '''We also have customer ids which starts with M in Receiver name, it indicates merchant
    this function will flag if there is a merchant in receiver end '''
    values = ['M']
    conditions = list(map(data['nameDest'].str.contains, values))
    data['merchant'] = np.select(conditions, '1', '0')


# Applying balance_diff function
balance_diff(paysim)

paysim['orig_diff'].value_counts()
paysim['dest_diff'].value_counts()
surge_indicator(paysim)
paysim['surge'].value_counts()
# Applying frequency_receiver function
frequency_receiver(paysim)
paysim['freq_dest'].value_counts()
# Creating a copy
paysim_1 = paysim.copy()

# Checking for balance in target
fig = go.Figure(data=[go.Pie(labels=['Not Fraud', 'Fraud'], values=paysim_1['isFraud'].value_counts())])
fig.show()
# Getting the max size
max_size = paysim_1['isFraud'].value_counts().min()

# Balancing the target label
lst = [paysim_1]
for class_index, group in paysim_1.groupby('isFraud'):
    lst.append(group.sample(max_size - len(group), replace=True))
paysim_1 = pd.concat(lst)

# Checking the balanced target
fig = go.Figure(data=[go.Pie(labels=['Not Fraud', 'Fraud'], values=paysim_1['isFraud'].value_counts())])
fig.show()

# One hot encoding
paysim_1 = pd.concat([paysim_1, pd.get_dummies(paysim_1['type'], prefix='type_')], axis=1)
paysim_1.drop(['type'], axis=1, inplace=True)

paysim_1.head()
# Splitting dependent and independent variable
paysim_2 = paysim_1.copy()
X = paysim_2.drop('isFraud', axis=1)
y = paysim_2['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=111)

# Standardizing the numerical columns
col_names = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
features_train = X_train[col_names]
features_test = X_test[col_names]
scaler = StandardScaler().fit(features_train.values)
features_train = scaler.transform(features_train.values)
features_test = scaler.transform(features_test.values)
X_train[col_names] = features_train
X_test[col_names] = features_test
# Tokenzation of customer name to get a unique id
tokenizer_org = tf.keras.preprocessing.text.Tokenizer()
tokenizer_org.fit_on_texts(X_train['nameOrig'])

tokenizer_dest = tf.keras.preprocessing.text.Tokenizer()
tokenizer_dest.fit_on_texts(X_train['nameDest'])

# Create tokenized customer lists
customers_train_org = tokenizer_org.texts_to_sequences(X_train['nameOrig'])
customers_test_org = tokenizer_org.texts_to_sequences(X_test['nameOrig'])

customers_train_dest = tokenizer_dest.texts_to_sequences(X_train['nameDest'])
customers_test_dest = tokenizer_dest.texts_to_sequences(X_test['nameDest'])

# Pad sequences
X_train['customers_org'] = tf.keras.preprocessing.sequence.pad_sequences(customers_train_org, maxlen=1)
X_test['customers_org'] = tf.keras.preprocessing.sequence.pad_sequences(customers_test_org, maxlen=1)

X_train['customers_dest'] = tf.keras.preprocessing.sequence.pad_sequences(customers_train_dest, maxlen=1)
X_test['customers_dest'] = tf.keras.preprocessing.sequence.pad_sequences(customers_test_dest, maxlen=1)

# Dropping unnecessary columns
X_train = X_train.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
X_train = X_train.reset_index(drop=True)

X_test = X_test.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
X_test = X_test.reset_index(drop=True)

# creating the objects
# logreg_cv = LogisticRegression(solver='liblinear',random_state=123)
dt_cv = DecisionTreeClassifier(random_state=123)
# knn_cv=KNeighborsClassifier()
# svc_cv=SVC(kernel='linear',random_state=123)
nb_cv = GaussianNB()
# rf_cv=RandomForestClassifier(random_state=123)
cv_dict = {0: 'Decision Tree', 1: 'Naive Bayes'}
cv_models = [dt_cv, nb_cv]

for i, model in enumerate(cv_models):
    print("{} Test Accuracy: {}".format(cv_dict[i],
                                        cross_val_score(model, X_test, y_test, cv=10, scoring='accuracy').mean()))

param_grid_nb = {
    'var_smoothing': np.logspace(0, -9, num=100)
}

nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
nbModel_grid.fit(X_train, y_train)
print(nbModel_grid.best_estimator_)

y_pred = nbModel_grid.predict(X_test)


# Function for Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    # conf_mat_norm= np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Predict with the selected best parameter


y_pred = nbModel_grid.predict(X_test)

# Plotting confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])