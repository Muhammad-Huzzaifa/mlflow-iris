from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE, RANDOM_STATE

def load_data():

    iris = load_iris()

    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    return X_train, X_test, y_train, y_test
