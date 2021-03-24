import numpy as np
from feats import get_mfcc
from getlist import get_list

def get_data(dirname):

    wavlist, train, test = get_list(dirname)
    train, test = sorted(train), sorted(test)
    train_data = [get_mfcc(name) for name in train]
    test_data = [get_mfcc(name) for name in test]
    label = np.arange(len(train))

    return list(train_data), list(test_data), label

def pad_data(data):

#    max_frame = np.max([len(x) for x in data])
#    pad_data = [np.pad(x,[(0,max_frame-len(x)),(0,0)]) for x in data]
    max_frame = 400
    pad_data_ = [np.pad(x,[(0,max_frame-len(x)),(0,0)]) for x in data]
    pad_data_ = np.array(pad_data_)
#    print(pad_data_.shape)
    pad_data = np.zeros((43,100,39))
    for k in range(43):
        for i in range(100):
            for j in range(39):
                pad_data[k][i][j] = (pad_data_[k][4*i][j] + pad_data_[k][4*i+1][j] + pad_data_[k][4*i+2][j] + pad_data_[k][4*i+3][j]) / 4
    return np.array(pad_data)

if __name__ == "__main__":

    train, test, label = get_data('new_data')
    for i, data in enumerate(train):
        print(i, data.shape)

    train = pad_data(train)
    for i, data in enumerate(train):
        print(i, data.shape)
