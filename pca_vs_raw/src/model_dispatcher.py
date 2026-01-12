
from sklearn import tree
from sklearn import ensemble

# dictionary that store machine learning model instances 
models = {
    "decision_tree_gini" : tree.DecisionTreeClassifier(criterion="gini", random_state=42),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy", random_state=42),
    "random_forest": ensemble.RandomForestClassifier(n_jobs=1, random_state=42),
    "extra_tree": ensemble.ExtraTreesClassifier(n_jobs=1, random_state=42)
}
