import numpy as np
from collections import defaultdict
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC, SVC


# C_VALUES = [0.01, 0.10, 1.00, 10.0, 100]
C_VALUES = [0.001, 0.01, 0.10, 1.0, 10.0, 100.0, 1000.0]


def cross_validation(X_train, y_train, n_folds=5):
    kf = KFold(len(X_train), n_folds)
    accuracy_dict = defaultdict(list)
    for i, (train_index, validation_index) in enumerate(kf):
        Xk_train, Xk_validation = X_train[train_index], X_train[validation_index]
        yk_train, yk_validation = y_train[train_index], y_train[validation_index]

        for C in C_VALUES:
            acc = calculate_accuracy(Xk_train, Xk_validation, yk_train, yk_validation, C)
            accuracy_dict[C].append(acc)
    return accuracy_dict


def calculate_accuracy(X_train, X_validation, y_train, y_validation, C=1.0):
    # Training
    # svm = LinearSVC(C=C)
    svm = SVC(C=10, gamma=C)
    svm.fit(X_train, y_train)

    # Validate
    y_pred = svm.predict(X_validation)

    # Calculate accuracy
    tp = np.sum(y_validation == y_pred)
    accuracy = float(tp) / len(y_validation)
    # print('accuracy: {:.3f}'.format(accuracy))
    return accuracy