import numpy as np

def transition_matrix(U):
    n = len(U) 
    UT = np.ones((n, n)) / n  
    return UT

def normalize_matrix(UT):
    return UT / np.sum(UT, axis=1, keepdims=True)

def ergodic_matrix(UN):
    size = len(UN)
    Q = UN - np.eye(size)
    ones = np.ones(size)
    Q = np.vstack((Q.T, ones))
    QTQ = np.dot(Q, Q.T)
    bQT = np.ones(size)
    return np.linalg.solve(QTQ, bQT)

def feature_selection(C, alpha):
    U = []
    for i in range(len(C)):
        U.append(C[i]) 
    UT = transition_matrix(U)
    UN = normalize_matrix(UT)
    Um = ergodic_matrix(UN)
    y = np.dot(Um, alpha)
    return y

def feature_selection_eig(Um):
    eigenvalues, eigenvectors = np.linalg.eig(Um)
    index = np.argmax(eigenvalues)
    y = eigenvectors[:, index]
    return y


C = [[gain ratio], [Infogain], [SU],[ReliefF],[MIFS]]

alpha = [0.1, 0.2, 0.7]
y_markov = feature_selection(C, alpha)
y_eig = feature_selection_eig(y_markov)
