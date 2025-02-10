import numpy as np

class Perceptron:
    """
    PERCEPTRON CLASSIFIER
    
    Parameters
    -----------
    eta: float  
    Learning rate (between 0.0 and 1.0)
    n_iter: int
    passes over the training dataset n times
    random_state: int
    random number generator seed for weight initialization
    
    Attributes
    -----------
    w_: 1D array
    weights after fitting
    b_: Scalar
    Bias unit after fitting
    erros_: list
    Number of misclassification after(updates) in each epoch
    """

    def __init__(self,eta = 0.01,n_iter = 200,random_state = 42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self,X,y):
        """
        Fit training data

        Parameters
        ------------
        X: {array-like},shape: [n_examples,n_features]
        Training vectors where n_examples is the number of examples and 
        n_features is the number of features
        y: array-like,shape: [n_examples]
        Target values

        Returns
        -------
        self: object

        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.error_ = []

        for _ in range(self.n_iter):
            errors = 0
            for Xi, target in zip(X,y):
                update = self.eta* (target - self.predict(Xi))
                self.w_ +=  update*Xi
                self.b_ += update
                errors += (update != 0)
            self.error_.append(errors)
        return self
    

    def net_input(self,X):
        """
        Calculate net input
        wTx + b
        """
        return np.dot(X,self.w_) + self.b_
    
    def predict(self,X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) > 0,1,0)
    
    def export_weights(self,filename = 'weights.txt'):
        mat = np.matrix(self.w_)
        with open(filename,'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt="%.2f")

    def export_bias(self,filename='bias.txt'):
        mat = np.matrix(self.b_)
        with open(filename,'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')

    


    