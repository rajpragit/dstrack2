import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


from sklearn import metrics


data = pd.read_csv("C:\\DataScience\\DataSets\\acute.csv", header=None)
data.columns=['temp','nausea','lpain','upushing','mpain','burethra','decision']
print(data.head())
print(data.shape)
#print(data.info())

tempMean=data.groupby('decision').mean()
print(tempMean)


sns.stripplot(x="decision", y="temp", data=data, jitter=True);
plt.show()

pd.crosstab(data.nausea,data.decision).plot(kind='bar')
plt.xlabel("Nausea")
plt.ylabel("Inflammation (yes) / Nephritis (no)")
plt.show()


pd.crosstab(data.lpain,data.decision).plot(kind='bar')
plt.ylabel("Inflammation (yes) / Nephritis (no)")
plt.xlabel("Lumbar pain")
plt.show()

pd.crosstab(data.upushing,data.decision).plot(kind='bar')
plt.ylabel("Inflammation (yes) / Nephritis (no)")
plt.xlabel("Urine pushing")
plt.show()

pd.crosstab(data.mpain,data.decision).plot(kind='bar')
plt.ylabel("Inflammation (yes) / Nephritis (no)")
plt.xlabel("Micturition pains")
plt.show()

pd.crosstab(data.burethra,data.decision).plot(kind='bar')
plt.ylabel("Inflammation (yes) / Nephritis (no)")
plt.xlabel("Burning of Urethra")
plt.show()


data['nauseaCode']= data.nausea.eq('yes').mul(1)
data['lpainCode']= data.lpain.eq('yes').mul(1)
data['upushingCode']= data.upushing.eq('yes').mul(1)
data['mpainCode']= data.mpain.eq('yes').mul(1)
data['burethraCode']= data.burethra.eq('yes').mul(1)
data['decisionCode']= data.decision.eq('yes').mul(1)

print(data.head())



#data.drop(data.columns[[1, 2, 3, 4, 5, 6]], axis=1, inplace=True)

data.drop(data.columns[[1, 2, 3, 4,5,6]], axis=1, inplace=True)

print(data.head())
sns.heatmap(data.corr())
plt.show()



X = data.iloc[:,1:]
y = data['decisionCode']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape)
print(X_test.shape)
#print(y)


X.head()



logreg = LogisticRegression()
#print(logreg)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


from sklearn import model_selection
#from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


