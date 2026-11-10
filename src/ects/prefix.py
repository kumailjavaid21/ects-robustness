"""Full-series classifier evaluated on growing prefixes without test-future leakage."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class PrefixClassifier:
    def fit(self,X,y):
        X=np.asarray(X,dtype=float);y=np.asarray(y)
        if X.ndim!=2 or not np.isfinite(X).all():raise ValueError('Expected finite univariate series')
        self.length=X.shape[1];self.fill=X.mean(axis=0);self.scaler=StandardScaler().fit(X)
        self.classifier=LogisticRegression(max_iter=2000,solver='lbfgs',random_state=42)
        self.classifier.fit(self.scaler.transform(X),y);self.classes_=self.classifier.classes_
        return self
    def probabilities(self,X,t):
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or X.shape[1]!=self.length or not 1<=t<=self.length:raise ValueError('Invalid prefix')
        # Do not read X[:,t:] at all: build the input from training means and observed prefix.
        masked=np.broadcast_to(self.fill,(len(X),self.length)).copy()
        masked[:,:t]=X[:,:t]
        return self.classifier.predict_proba(self.scaler.transform(masked))

def prefix_times(length,fractions):
    return np.array([max(1,min(length,int(np.ceil(length*f)))) for f in fractions],dtype=int)

def probabilities_over_time(model,X,times):
    return np.stack([model.probabilities(X,int(t)) for t in times],axis=0)

def fixed_predictions(probabilities,classes,times,index):
    return classes[probabilities[index].argmax(axis=1)],np.full(probabilities.shape[1],times[index],dtype=int)

def threshold_predictions(probabilities,classes,times,threshold):
    confident=probabilities.max(axis=2)>=threshold
    # The final observation is always forced, including thresholds never reached.
    confident[-1]=True
    indices=confident.argmax(axis=0)
    cases=np.arange(probabilities.shape[1])
    return classes[probabilities[indices,cases].argmax(axis=1)],times[indices]
