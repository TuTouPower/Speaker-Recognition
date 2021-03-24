import torch
from model import Net
from dataset import get_data, pad_data

device = 'cuda' # 'cpu'

def train_network(net, data, label):

    net.train()
    predict = net(data) # 43, 43

    net.optimizer.zero_grad()
    net.loss(predict, label).backward()
    net.optimizer.step()

def test_network(net, data, label):

    net.eval()
    with torch.no_grad():
        predict_label = net(data).argmax(1)
        acc = torch.sum(label==predict_label).float()/len(label)
        return acc.to('cpu').numpy()

if __name__ == "__main__":

    net = Net(39, 43).to(device)
    net.loss = torch.nn.CrossEntropyLoss().to(device)
    net.optimizer = torch.optim.Adam(net.parameters(), lr=0.0008)
    
    train, test, label = get_data('../data')
    train, test = pad_data(train), pad_data(test)

    train = torch.from_numpy(train).float().permute(0,2,1).to(device)
    test = torch.from_numpy(test).float().permute(0,2,1).to(device)

    label = torch.from_numpy(label).long().to(device)

    print(train.shape, test.shape, label.shape)
    
    for i in range(666):
        train_network(net, train, label)
        train_acc = test_network(net, train, label)
        test_acc = test_network(net, test, label)
        print('%03d: train=%.3f test=%.3f'%(i+1, train_acc, test_acc))

    torch.save(net,'model.pkl')
    mdl=torch.load('model.pkl')

    train_acc = test_network(mdl, train, label)
    test_acc = test_network(mdl, test, label)

    print('train=%.3f test=%.3f'%(train_acc, test_acc))
