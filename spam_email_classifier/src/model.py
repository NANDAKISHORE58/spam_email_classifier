import numpy as np

class NumpyNB:
    def __init__(self):
        self.classes = None
        self.means = None
        self.vars = None
        self.priors = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        self.means = np.zeros((len(self.classes), n_features))
        self.vars = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))
        
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[i] = np.mean(X_c, axis=0)
            self.vars[i] = np.var(X_c, axis=0) + 1e-4
            self.priors[i] = X_c.shape[0] / X.shape[0]
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, len(self.classes)))
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i] + 1e-9)
            coef = -0.5 / self.vars[i]
            log_cond = np.sum(coef * (X - self.means[i])**2, axis=1)
            proba[:, i] = prior + log_cond
        proba -= np.max(proba, axis=1, keepdims=True)
        proba = np.exp(proba)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba
    
    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]
