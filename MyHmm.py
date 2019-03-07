from hmmlearn import hmm


class MyHmm(object):
    
    def __init__(self, n_components=10):
        self.models = {}
        self.n_components = n_components
    
    def fit(self, Xs, ys):
        for X, label in zip(Xs, ys):
            if label not in self.models:
                self.models[label] = hmm.GaussianHMM(n_components=self.n_components)
            self.models[label].fit(X)
            
    def predict(self, Xs):
        return map(self.best_score_label, Xs)

    def best_score_label(self, X):
        scores = dict((label, model.score_samples(X)[0]) for label, model in self.models.items())
        sorted_scores = sorted(scores.items(), key=lambda (label, score): score, reverse=True)
        return sorted_scores[0][0]