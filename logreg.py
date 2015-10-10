import numpy as np

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None

    def computeCost(self, theta, X, y, regLambda):
        hypo = 1/(1+(np.exp(-np.dot(X,theta))))
        y_1 = np.dot(y.T,np.log(hypo))
        y_0 = np.dot(1-y.T,np.log(1-hypo))        
        reg = regLambda * np.sum(np.power(theta,2),axis=0)
        cost = -(y_1+y_0)+reg
        cost = np.asscalar(cost)
        return cost

    def computeGradient(self, theta, X, y, regLambda):
        counter = 0
        theta_new = 5
        while True:
            hypo = 1/(1+(np.exp(-np.dot(X,theta))))
            diff = hypo - y
            desc = np.dot(X.T,diff) + (regLambda * theta)
            theta_old = theta
            theta_new = theta_old - ((self.alpha/X.shape[0]) * desc)
            theta = theta_new
            counter = counter + 1
            if np.sqrt(np.sum(np.power((theta_new-theta_old),2))) < self.epsilon or counter == self.maxNumIters:
                break
        return theta_new  

    def fit(self, X, y):
        q = X.shape[0]    
        X_1 = np.c_[np.ones((q,1)),X]
        p = X_1.shape[1]
        thetaEmpty = np.zeros((p,1))
        self.theta = self.computeGradient(thetaEmpty, X_1, y, regLambda = self.regLambda)
        
    def predict(self, X):
        m = X.shape[0]    
        X_pred = np.c_[np.ones((m,1)),X]
        predictions = np.array(1/(1+(np.exp(-np.dot(X_pred,self.theta)))))
        return predictions