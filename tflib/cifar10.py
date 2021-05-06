import numpy as np

import os
import urllib
import gzip
import pickle as pickle

def unpickle(file):
    # print(file)
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    # print(dict.keys())
    fo.close()
    return dict[b'data']

def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    for filename in filenames:
        all_data.append(unpickle(data_dir + '/' + filename))

    images = np.concatenate(all_data, axis=0)

    def get_epoch():
        np.random.shuffle(images)

        for i in range(int(len(images) / batch_size)):
            yield np.copy(images[int(i*batch_size):int((i+1)*batch_size)])

    return get_epoch


def load(batch_size, data_dir):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir), 
        cifar_generator(['test_batch'], batch_size, data_dir)
    )
