from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier


# Pick the Model Estimator
ClassifierDict = {
    model.__class__.__name__: model
    for model in [
        RandomForestClassifier(bootstrap=True),
        BalancedRandomForestClassifier(bootstrap=True),
    ]
}
