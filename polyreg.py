import numpy as np

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        self.regLambda = regLambda
        self.degree = degree

    def polyfeatures(self, X, degree):
        X_feat = X
        for i in range(2,degree+1):            
            new_d = np.power(X_feat[:,0],i)
            X_feat = np.c_[X_feat,new_d]
        return X_feat
            
    def fit(self, X, y):
        X_fit = np.matrix(X).T
        y_fit = np.matrix(y).T
        X_poly = self.polyfeatures(X_fit,self.degree)
        
        self.X_means = np.mean(X_poly,axis=0).T
        self.X_std = np.std(X_poly,axis=0).T
        for i,j,k in zip(self.X_means,self.X_std,range(0,X_poly.shape[1])):
            mean = i
            std = j
            for l in range(0,X_poly.shape[0]):
                X_poly[l,k] = (X_poly[l,k] - mean) / std 
        
        X_poly_full = np.c_[np.ones([X_poly.shape[0], 1]), X_poly]        
        
        regMatrix = self.regLambda * np.eye(X_poly_full.shape[1])
        regMatrix[0,0] = 0

        self.theta_vec = np.linalg.pinv(X_poly_full.T.dot(X_poly_full) + regMatrix).dot(X_poly_full.T).dot(y_fit)       
        
    def predict(self, X):
        xpoints_pred = np.matrix(X).T
        xpoints_poly = self.polyfeatures(xpoints_pred,self.degree)
        
        xpoints_poly_means = self.X_means
        xpoints_poly_std = self.X_std
        for i,j,k in zip(xpoints_poly_means,xpoints_poly_std,range(0,xpoints_poly.shape[1])):
            mean = i
            std = j
            for l in range(0,xpoints_poly.shape[0]):
                xpoints_poly[l,k] = (xpoints_poly[l,k] - mean) / std         
        
        xpoints_poly_full = np.c_[np.ones([xpoints_poly.shape[0], 1]), xpoints_poly]   
        
        pred = xpoints_poly_full.dot(self.theta_vec)
        return pred
        
#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------

def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    n = len(Xtrain)
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    for i in range(0,n):
        X_sets = [Xtrain[0:(i+1)],Xtest]
        y_sets = [Ytrain[0:(i+1)],Ytest]    
        for j in range(0,2):
            X_fit = np.matrix(X_sets[j]).T
            y_fit = np.matrix(y_sets[j]).T
            for k in range(2,degree+1):            
                new_d = np.power(X_fit[:,0],k)
                X_fit = np.c_[X_fit,new_d]   
            X_poly = X_fit
            length = len(X_poly)
            if j == 0:
                X_means = np.mean(X_poly,axis=0).T
                X_std = np.std(X_poly,axis=0).T
                if np.sum(X_std) != 0:
                    for m,s,r in zip(X_means,X_std,range(0,X_poly.shape[1])):
                        for w in range(0,X_poly.shape[0]):
                            X_poly[w,r] = (X_poly[w,r] - m) / s 
                X_poly_full = np.c_[np.ones([X_poly.shape[0], 1]), X_poly]        
                regMatrix = regLambda * np.eye(X_poly_full.shape[1])
                regMatrix[0,0] = 0
                theta_vec = np.linalg.pinv(X_poly_full.T.dot(X_poly_full) + regMatrix).dot(X_poly_full.T).dot(y_fit)       
                errorTrain[i] = np.sum(np.power(X_poly_full.dot(theta_vec) - y_fit,2)) / length
            if j == 1:
                if np.sum(X_std) != 0:
                    for z,y,r in zip(X_means,X_std,range(0,X_poly.shape[1])):
                        for w in range(0,X_poly.shape[0]):
                            X_poly[w,r] = (X_poly[w,r] - m) / s
                X_poly_full = np.c_[np.ones([X_poly.shape[0], 1]), X_poly]
                errorTest[i] = np.sum(np.power(X_poly_full.dot(theta_vec) - y_fit,2)) / length
    return (errorTrain, errorTest)