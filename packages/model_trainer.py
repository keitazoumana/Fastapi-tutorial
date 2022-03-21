from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

def run_model_training(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    return clf
