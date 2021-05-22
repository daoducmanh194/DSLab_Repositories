import re
import os
from os.path import *
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def gather_20newsgroups_data():
    # read data and gather it
    path = "/home/genkibaskervillge/Documents/MachineLearning/datasets/20news-bydate/"
    dirs = [path + dir_name + '/' for dir_name in os.listdir(path) if not isfile(path + dir_name)]
    train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])
    list_newsgroups = [newgroup for newgroup in os.listdir(train_dir)]
    list_newsgroups.sort()

    # data collection
    with open("/home/genkibaskervillge/Documents/MachineLearning/datasets/20news-bydate/stop_words.txt",
              encoding="utf8", errors='ignore') as f:
        stop_words = f.read().splitlines()

    def collect_data_from(parent_dir, newsgroup_list):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            label = group_id
            dir_path = parent_dir + '/' + newsgroup + '/'
            files = [(filename, dir_path + filename) for filename in os.listdir(dir_path) if
                     isfile(dir_path + filename)]
            files.sort()
            for filename, filepath in files:
                with open(filepath, errors='ignore') as sf:
                    text = sf.read().lower()
                    # remove stop word then stem remain word
                    words = [stemmer.stem(word) for word in re.split(r'\W+', text) if word not in stop_words]
                    # combine remain words
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content)
        return data

    train_data = collect_data_from(
        parent_dir=train_dir,
        newsgroup_list=list_newsgroups
    )

    test_data = collect_data_from(
        parent_dir=test_dir,
        newsgroup_list=list_newsgroups
    )

    # note to file
    full_data = train_data + test_data
    with open("/home/genkibaskervillge/Documents/MachineLearning/datasets/20news-bydate/20news-train-processed.txt",
              'w') as f:
        f.write(join('\n'.join(train_data)))

    with open("/home/genkibaskervillge/Documents/MachineLearning/datasets/20news-bydate/20news-test-processed.txt",
              'w') as f:
        f.write(join('\n'.join(test_data)))

    with open("/home/genkibaskervillge/Documents/MachineLearning/datasets/20news-bydate/20news-full-processed.txt",
              'w') as f:
        f.write(join('\n'.join(full_data)))


gather_20newsgroups_data()
