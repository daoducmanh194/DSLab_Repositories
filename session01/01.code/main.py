from __future__ import division, print_function, unicode_literals
import numpy as np

# Read raw file and export important rows
import pandas as pd

rawFilePath = "/home/genkibaskervillge/x28.txt"
with open(rawFilePath) as f:
    contentList = f.readlines()
print(contentList)
print("\n")
# Only store data contains necessary field
with open("/home/genkibaskervillge/death-rates-data.txt", 'w') as sf:
    sf.writelines(contentList[(len(contentList) - 61):])
# Change from data to vector
# X = []  # set for store 15 features
# y = []  # set for store labels


def get_data(path):
    """"
    X = []  # set for store 15 features
    y = []  # set for store labels
    with open(path) as rf:
        allContent = rf.readlines()
        for j in range(0, 60):
            content = allContent[j]
            # print(allContent[j])
            listContent = content.split()
            for i in range(0, len(listContent)):
                listContent[i] = float(listContent[i])
            # npContent = np.array(listContent)
            X.append(listContent[1:-2])
            y.append(listContent[-1])
            # print(npContent)
    return X, y
    """
    data = pd.read_csv(path, header=None, sep=r'\s+')
    print(data[0:5])
    data.drop(data.columns[0], axis=1, inplace=True)
    data = np.array(data)
    X = data[:,:15]
    y = data[:,15:]
    return X, y


# X, y = get_data('/home/genkibaskervillge/death-rates-data.txt')
# print(X.shape)
# print(X[0:5])


"""""
X, Y = get_data('/home/genkibaskervillge/death-rates-data.txt')
print(X)
print(y)
print(len(y))
"""""


# normalization
def normalize_and_add_ones(X):
    X = np.array(X)
    # print(X)
    # print(X.shape)
    X_max = np.array([[np.amax(X[:, column_id])
                       for column_id in range(X.shape[1])]
                      for _ in range(X.shape[0])])
    # print(X_max)
    # print("\n", X_max.shape)
    X_min = np.array([[np.amin(X[:, column_id])
                       for column_id in range(X.shape[1])]
                      for _ in range(X.shape[0])])
    # print(X_min)
    # print("\n", X_min.shape)
    X_normalized = (X - X_min) / (X_max - X_min)
    ones = np.array([[1] for _ in range(X_normalized.shape[0])])
    # print(np.column_stack((ones, X_normalized)))
    return np.column_stack((ones, X_normalized))


# model
class RidgeRegression:
    def __init__(self):
        return

    def fit(self, X_train, Y_train, LAMBDA):
        assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]

        W = np.linalg.inv(X_train.transpose().dot(X_train) + LAMBDA * np.identity(X_train.shape[1])) \
            .dot(X_train.transpose()).dot(Y_train)
        return W

    def predict(self, W, X_new):
        X_new = np.array(X_new)
        Y_new = X_new.dot(W)
        return Y_new

    def compute_RSS(self, Y_new, Y_predicted):
        loss = 1. / Y_new.shape[0] * np.sum((Y_new - Y_predicted) ** 2)
        return loss

    def get_the_best_LAMBDA(self, X_train, Y_train):
        def cross_validation(num_folds, LAMBDA):
            row_ids = np.array(range(X_train.shape[0]))
            # print(row_ids)
            # np.split() requires equal divisions
            valid_ids = np.split(row_ids[:len(row_ids) - len(row_ids) % num_folds], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds:])
            # print(valid_ids)
            train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
            # print(train_ids)
            aver_RSS = 0
            for i in range(num_folds):
                valid_part = {'X': X_train[valid_ids[i]], 'Y': Y_train[valid_ids[i]]}
                train_part = {'X': X_train[train_ids[i]], 'Y': Y_train[train_ids[i]]}
                W = self.fit(train_part['X'], train_part['Y'], LAMBDA)
                Y_predicted = self.predict(W, valid_part['X'])
                aver_RSS += self.compute_RSS(valid_part['Y'], Y_predicted)
            return aver_RSS / num_folds

        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
            for current_LAMBDA in LAMBDA_values:
                aver_RSS = cross_validation(num_folds=5, LAMBDA=current_LAMBDA)
                if aver_RSS < minimum_RSS:
                    best_LAMBDA = current_LAMBDA
                    minimum_RSS = aver_RSS
            return best_LAMBDA, minimum_RSS

        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=0, minimum_RSS=1000 ** 2, LAMBDA_values=range(50))
        # [0, 1, 2, ..., 50]
        LAMBDA_values = [k * 1. / 1000 for k in range(max(0, (best_LAMBDA - 1) * 1000), (best_LAMBDA + 1) * 1000, 1)]
        # step size = 0.001
        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=best_LAMBDA, minimum_RSS=minimum_RSS,
                                              LAMBDA_values=LAMBDA_values)
        return best_LAMBDA


if __name__ == '__main__':
    X, Y = get_data(path='/home/genkibaskervillge/death-rates-data.txt')
    # normalization
    X = normalize_and_add_ones(X)
    print(X.shape)
    X_train, Y_train = X[:50], Y[:50]
    X_test, Y_test = X[50:], Y[50:]

    ridge_regression = RidgeRegression()
    best_LAMBDA = ridge_regression.get_the_best_LAMBDA(X_train, Y_train)
    print('Best LAMBDA:', best_LAMBDA)
    W_learned = ridge_regression.fit(X_train=X_train, Y_train=Y_train, LAMBDA=best_LAMBDA)
    # print(W_learned)
    Y_predicted = ridge_regression.predict(W=W_learned, X_new=X_test)
    print(ridge_regression.compute_RSS(Y_new=Y_test, Y_predicted=Y_predicted))