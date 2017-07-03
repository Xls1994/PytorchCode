# -*- coding: utf-8 -*-
'''
author:yangyl

'''
import torch
import torch.nn as nn
import torch.optim as optim
from cnn import Net, CONFIG
from torch.autograd import Variable
import time
from utils import loadData
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.backends.cudnn.benchmark = True
print "Setting seed..."
seed = 1234
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
(train_set_x,train_set_y),(test_set_x,test_set_y),W =loadData('./mr.p')

# setup CNN model
CONFIG["vocab_size"] = W.shape[0]
CONFIG["num_classes"] = 2
model = Net()

if torch.cuda.is_available():
    print "CUDA is available on this machine. Moving model to GPU..."
    model.cuda()
print model

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
n_epoch =2
step =0
batch_size =10
total_size =train_set_x.shape[0]

def _get_variable(tensor, volatile=False):
    data = Variable(tensor, volatile=volatile)
    if torch.cuda.is_available():
        data = data.cuda()
    return data

for epoch in xrange(1, n_epoch + 1):
    print "Epoch#{}".format(epoch)

    step += 1
    start_time = time.time()

    optimizer.zero_grad()
    samples =np.random.choice(total_size,batch_size)
    print samples
    outputs = model(_get_variable(torch.from_numpy(train_set_x[samples])))
    # print outputs
    loss = criterion(outputs, _get_variable(torch.from_numpy(train_set_y[samples])))
    print 'loss: ',loss.data
    loss.backward()
    optimizer.step()
    duration = time.time() - start_time


if __name__ == '__main__':
    pass