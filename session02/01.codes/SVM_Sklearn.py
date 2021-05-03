import numpy as np
from sklearn.svm import LinearSVC


class apply_svm:
    def __init__(self):
        return

    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split()
            for index_tfidf in indices_tfidfs:
                index = int(index_tfidf.split(':')[0])
                tfidf = float(index_tfidf.split(':')[1])
                r_d[index] = tfidf
            return np.array(r_d)

        with open(data_path) as f:
            d_lines = f.read().splitlines()
        with open('/home/genkibaskervillge/Documents/MachineLearning/datasets/'
                  '20news-bydate/words_idfs.txt') as f:
            vocab_size = len(f.read().splitlines())

        data = []
        label_list = []

        for data_id, d in enumerate(d_lines):
            features = d.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)

            data.append(r_d)
            label_list.append(label)
        return np.array(data), np.array(label_list)

    def compute_accuracy(self, predicted_y, expected_y):
        matches = np.equal(predicted_y, expected_y)
        accuracy = np.sum(matches.astype(float)) / expected_y.size
        return accuracy

    def classifying_with_linear_SVMs(self):
        train_X, train_y = self.load_data(data_path='/home/genkibaskervillge/Documents/MachineLearning/datasets/'
                                                  '20news-bydate/20news-train-tf-idf.txt')
        classifier = LinearSVC(
            C=1.0,  # penalty coefficient = 1 / r
            tol=0.001,   # stopping criteria
            verbose=True
        )
        classifier.fit(train_X, train_y)

        test_X, test_y = self.load_data(data_path='/home/genkibaskervillge/Documents/MachineLearning/datasets/'
                                                  '20news-bydate/20news-test-tf-idf.txt')
        predicted_y = classifier.predict(test_X)
        accuracy = self.compute_accuracy(predicted_y=predicted_y, expected_y=test_y)
        print('Accuracy: \n', accuracy)

        from sklearn import svm
        for kernel in ('sigmoid', 'poly', 'rbf'):
            clf = svm.SVC(kernel=kernel, gamma=1, coef0=1)
            clf.fit(train_X, train_y)
            pred_y = clf.predict(test_X)
            acc = self.compute_accuracy(predicted_y=pred_y, expected_y=test_y)
            print('Accuracy ' + kernel + acc + '\n')


using_SVM = apply_svm()
using_SVM.classifying_with_linear_SVMs()
