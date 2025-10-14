'''
1404-07-22,
Mohammad Kadkhodaei,
https://www.geeksforgeeks.org/python/kalman-filter-in-python/
'''

import numpy as np

class KalmanFilter:
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
    def predict(self, u):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x

F = np.array([[1, 1], [0, 1]])
B = np.array([[0.5], [1]])
H = np.array([[1, 0]])
Q = np.array([[1, 0], [0, 1]])
R = np.array([[1]])

x0 = np.array([[0], [1]])
P0 = np.array([[1, 0], [0, 1]])

kf = KalmanFilter(F, B, H, Q, R, x0, P0)

u = np.array([[1]])
z = np.array([[1]])

predicted_state = kf.predict(u)
print("Predicted state:\n", predicted_state)

updated_state = kf.update(z)
print("Updated state:\n", updated_state)
