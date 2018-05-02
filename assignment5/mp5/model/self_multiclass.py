import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        bsvm = {}
        for i in range(10):
            c = y == i
            classifier = svm.LinearSVC(random_state=12345)
            classifier.fit(X, c)
            bsvm[i] = classifier
        return bsvm

    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        bsvm = {}

        for i in range(10):
            for j in range(i + 1, 10, 1):
                # Array of samples and features with
                # samples for the current 1v1 classification
                labels = []
                features = []

                # Iterate through y array to find samples that apply to i and j
                for k in range(len(y)):
                    if(y[k] == i or y[k] == j):
                        labels.append(y[k])
                        features.append(X[k])

                # 1 if equal to the first class, 0 if equal to second class
                for p in range(len(labels)):
                    if(labels[p] == i):
                        labels[p] = 1
                    else:
                        labels[p] = 0

                features = np.array(features)
                labels = np.array(labels)
                classifier = svm.LinearSVC(random_state=12345)
                classifier.fit(features, labels)
                bsvm[(i, j)] = classifier

        return bsvm

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = [[0 for i in range(10)] for j in range(len(X))]
        for l, classifier in self.binary_svm.items():
            prediction = classifier.predict(X)
            dec = classifier.decision_function(X)

            for i in range(len(dec)):
                scores[i][l] += dec[i]

        scores = np.array(scores)
        return scores

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = [[0 for i in range(10)] for j in range(len(X))]
        for l, classifier in self.binary_svm.items():
            prediction = classifier.predict(X)
            for i in range(len(prediction)):
                # Vote for label l[0] to this sample by this classifier
                if(prediction[i]):
                    scores[i][l[0]] += 1

                else:
                    scores[i][l[1]] += 1
        scores = np.array(scores)
        return scores

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        loss = (C/2) * np.sum(W**2)
        scores = np.matmul(W, np.transpose(X))
        correct_scores = []
        for i in range(len(scores[0])):
            correct_scores.append(scores[y[i]][i])
            
            # Substract -1 in the original score to account for delta_{j,y} later
            scores[y[i]][i] -= 1
        correct_scores = np.array(correct_scores)

        # Calculate the max for each row
        scores = np.amax(scores, axis=0)
        margin = 1 + scores - correct_scores
        for i in range(len(margin)):
            if(margin[i]>0):
                loss+=margin[i]

        return loss

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        dW = np.zeros(W.shape)
        scores = np.matmul(W, np.transpose(X))

        correct_scores = []
        for i in range(len(scores[0])):
            correct_scores.append(scores[y[i]][i])
            
            # Substract -1 in the original score to account for delta_{j,y} later
            scores[y[i]][i] -= 1
        correct_scores = np.array(correct_scores)

        # Calculate the max for each row
        max_indices = np.argmax(scores, axis=0)
        scores = np.amax(scores, axis=0)
        margin = 1 + scores - correct_scores
        for i in range(len(margin)):
            if (margin[i] > 0):
                dW[y[i], :] -= X[i]
                dW[max_indices[i], :] += X[i]
        return C*W + dW
        
        
        
