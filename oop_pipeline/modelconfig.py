import json
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Models dictionnary
MODELS = {
    "lr": {
        "name": "Logistic Regression",
        "api_model_code": "lr",
        "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        "model": "LogisticRegression",
        "parameters": {
            "C": 1.0,              # Regularization strength
            "max_iter": 500,       # Maximum iterations
            "penalty": "l2"        # Regularization norm
        }
    },
    "dt": {
        "name": "Decision Tree",
        "api_model_code": "dt",
        "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
        "model": "DecisionTreeClassifier",
        "parameters": {
            "max_depth": None,     # Maximum depth of the tree
            "min_samples_split": 2,# Minimum number of samples to split an internal node
            "min_samples_leaf": 1  # Minimum number of samples in a leaf node
        }
    },
    "knn": {
        "name": "K Nearest Neighbors",
        "api_model_code": "knn",
        "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html",
        "model": "KNeighborsClassifier",
        "parameters": {
            "n_neighbors": 5,      # Number of neighbors to use
            "weights": "uniform",  # Weight function used in prediction
            "algorithm": "auto"    # Algorithm used to compute the nearest neighbors
        }
    },
    "svc": {
        "name": "Support Vector Classifier",
        "api_model_code": "svc",
        "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
        "model": "SVC",
        "parameters": {
            "C": 1.0,              # Regularization parameter
            "kernel": "rbf",       # Specifies the kernel type
            "gamma": "scale"       # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
        }
    },
    "rf": {
        "name": "Random Forest Classifier",
        "api_model_code": "rf",
        "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
        "model": "RandomForestClassifier",
        "parameters": {
            "n_estimators": 100,   # Number of trees in the forest
            "max_depth": None,     # Maximum depth of the tree
            "min_samples_split": 2 # Minimum number of samples required to split a node
        }
    },
    "gb": {
        "name": "Gradient Boosting Classifier",
        "api_model_code": "gb",
        "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
        "model": "GradientBoostingClassifier",
        "parameters": {
            "n_estimators": 100,   # Number of boosting stages to perform
            "learning_rate": 0.1,  # Learning rate shrinks the contribution of each tree
            "max_depth": 3         # Maximum depth of the individual regression estimators
        }
    },
    "ada": {
        "name": "AdaBoost Classifier",
        "api_model_code": "ada",
        "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",
        "model": "AdaBoostClassifier",
        "parameters": {
            "n_estimators": 50,    # Maximum number of estimators at which boosting is terminated
            "learning_rate": 1.0,  # Learning rate shrinks the contribution of each classifier
            "algorithm": "SAMME.R" # Algorithm to use for weight updating
        }
    },
    "gnb": {
        "name": "Gaussian Naive Bayes",
        "api_model_code": "gnb",
        "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html",
        "model": "GaussianNB",
        "parameters": {
            "var_smoothing": 1e-9  # Portion of the largest variance of all features that is added to variances for calculation stability
        }
    }
}

# A class to encapsulate model configuration and allow instantiation.
class ModelConfig:
    def __init__(self, config):
        self.name = config["name"]
        self.api_model_code = config["api_model_code"]
        self.documentation = config["documentation"]
        self.model_class_name = config["model"]
        self.parameters = config.get("parameters", {})

    def instantiate(self):
        # Mapping from string names to actual scikit-learn classes.
        mapping = {
            "LogisticRegression": LogisticRegression,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "KNeighborsClassifier": KNeighborsClassifier,
            "SVC": SVC,
            "RandomForestClassifier": RandomForestClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "AdaBoostClassifier": AdaBoostClassifier,
            "GaussianNB": GaussianNB
        }
        model_class = mapping.get(self.model_class_name)
        if model_class is None:
            raise ValueError(f"Model class {self.model_class_name} not found in mapping.")
        return model_class(**self.parameters)

    def update_parameters(self, new_params):
        """Update parameters with a dictionary of new values."""
        self.parameters.update(new_params)

    def get_parameters(self):
        """Return the current parameters."""
        return self.parameters

    def to_dict(self):
        """Export the configuration as a dictionary."""
        return {
            "name": self.name,
            "api_model_code": self.api_model_code,
            "documentation": self.documentation,
            "model": self.model_class_name,
            "parameters": self.parameters
        }

# Trainer that uses the ModelConfig object.
class ModelTrainer:
    def __init__(self, model_config):
        """
        :param model_config: An instance of ModelConfig.
        """
        self.model_config = model_config
        self.model = None

    def train_model(self, X, y):
        """
        Split the data, instantiate a model from the configuration,
        train it, and report test accuracy.
        
        :param X: Features DataFrame.
        :param y: Target Series.
        :return: The trained model pipeline.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        # Instantiate the classifier from our configuration
        clf = self.model_config.instantiate()
        # Build a pipeline (if needed, you could add scaling, encoding, etc.)
        self.model = make_pipeline(clf)
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        print(f"Model {self.model_config.name} trained with test accuracy: {accuracy}")
        return self.model

# --- Example usage ---
# Suppose you want to select a model based on a parameter (e.g., 'lr' for logistic regression).
selected_model_code = "lr"
model_config = ModelConfig(MODELS[selected_model_code])

# Optionally, update parameters (for instance, if you want to experiment with a lower C)
# model_config.update_parameters({"C": 0.5, "max_iter": 300})

#Create a trainer and run the training process.
trainer = ModelTrainer(model_config)
trained_model = trainer.train_model(X, y)
