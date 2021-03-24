import os

def get_list(dirname):
    files = os.listdir(dirname)

    files = filter(lambda x: x.endswith('.wav'), files)
    files = ['%s/%s'%(dirname,name) for name in files]

    train = filter(lambda x: x.endswith('1-16K.wav'), files)
    test = filter(lambda x: x.endswith('2-16K.wav'), files)

    return list(files), list(train), list(test)

if __name__ == "__main__":

    wavlist, train, test = get_list('new_data')
    
    for i,name in enumerate(test):
        print(i, name)
        