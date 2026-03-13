from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(X_train, y_train, **kwargs):

    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    
    return model

def train_random_forest(X_train, y_train, **kwargs):

    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    
    return model
