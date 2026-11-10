"""A TEASER base wrapper that computes safety-model CV on the fitted prefix."""
import numpy as np
from sklearn.model_selection import cross_val_predict
from aeon.classification.sklearn import SklearnClassifierWrapper

class PrefixCVClassifier(SklearnClassifierWrapper):
    # aeon TEASER otherwise passes full series to generic estimator cross-validation.
    _save_train_predictions=True
    def _fit(self,X,y):
        self._prefix_training_data=np.array(X,copy=True)
        return super()._fit(X,y)
    def _get_train_probs(self,X,y):
        _,counts=np.unique(y,return_counts=True)
        cv=min(5,int(counts.min()))
        if cv<2:raise ValueError('TEASER prefix CV requires at least 2 examples per class')
        return cross_val_predict(self.classifier,self._prefix_training_data,y=y,cv=cv,method='predict_proba',n_jobs=1)
