"""Aeon TEASER adapter reusing checkpoint MiniRocket features and classifiers."""
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDOneClassSVM
from aeon.classification import BaseClassifier
from aeon.classification.early_classification import TEASER
from .rocket_v2 import make_classifier,metrics
REGISTRY={}

class CachedRocketClassifier(BaseClassifier):
    _tags={'X_inner_type':'numpy2D'}
    _save_train_predictions=True
    def __init__(self,random_state=None):
        self.random_state=random_state
        super().__init__()
    def _fit(self,X,y):
        self._entry=REGISTRY[X.shape[1]]
        if not np.array_equal(X,self._entry['fit_input']) or not np.array_equal(y,self._entry['fit_y']):
            raise ValueError('TEASER fit data do not match cached checkpoint')
        return self
    def _get_train_probs(self,X,y):
        # The transform is unsupervised and shared across folds; the logistic fit is OOF.
        counts=np.unique(y,return_counts=True)[1];folds=min(3,int(counts.min()))
        if folds<2:raise ValueError('TEASER requires at least two fit examples per class')
        return cross_val_predict(make_classifier(),self._entry['fit_features'],y,cv=folds,method='predict_proba',n_jobs=1)
    def _predict_proba(self,X):
        if np.array_equal(X,self._entry['test_input']):
            return self._entry['test_probs']
        x=np.ascontiguousarray(X[:,None,:],dtype=np.float32)
        return self._entry['classifier'].predict_proba(self._entry['transform'].transform(x))
    def _predict(self,X):
        return self.classes_[self._predict_proba(X).argmax(axis=1)]

def evaluate_teaser(fit,y,test,z,times,registry):
    REGISTRY.clear();REGISTRY.update(registry)
    model=TEASER(estimator=CachedRocketClassifier(random_state=42),
        one_class_classifier=SGDOneClassSVM(nu=.1,max_iter=500,tol=.001,random_state=42),
        one_class_param_grid={'nu':[.1]},classification_points=list(map(int,times)),n_jobs=1,random_state=42)
    model.fit(fit[:,None,:],y)
    prediction,accepted=model.predict(test[:,None,:])
    if not np.all(accepted):raise ValueError('TEASER has undecided test samples at full length')
    decision=np.asarray(model.state_info[:,3],dtype=int)
    out=metrics(z,prediction,decision,test.shape[1]);out.update({'baseline':'TEASER','setting':'aeon'})
    return out
