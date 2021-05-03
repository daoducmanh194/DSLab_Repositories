import numpy as np


def load_data(data_path):
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
    with open('/home/genkibaskervillge/Documents/MachineLearning/datasets/20news-bydate/words_idfs.txt') as f:
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


def cluster_with_Kmeans():
    data, labels = load_data("/home/genkibaskervillge/Documents/MachineLearning/"
                             "datasets/20news-bydate/data_tf_idf.txt")
    from sklearn.cluster import KMeans
    from scipy.sparse import csr_matrix
    X = csr_matrix(data)
    print("======")
    kmeans = KMeans(
        n_clusters=20,
        init='random',
        n_init=5,
        tol=1e-8,
        random_state=2021
    ).fit(X)
    pred_labels = kmeans.labels_
    from sklearn.metrics.cluster import completeness_score
    print("Completeness score: {}".format(completeness_score(labels, pred_labels)))


cluster_with_Kmeans()
# Completeness score: 0.5148094073408562
