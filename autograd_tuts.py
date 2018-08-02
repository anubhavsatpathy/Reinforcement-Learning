import torch
import numpy as np
import torch.utils.data

# Autograd is a library used to differentiate various DL functions used within the layer
# Each torch tensor comes with a flag requires_grad

x = torch.zeros(size = [5,5])
print(x.requires_grad)

# As noted above the default value for user-created tensors is false
# This means, when backward method is called on any tensor in the graph, the grad w.r.t these tensors are not computed

y = torch.ones(size = [5,5])
z = x + y
print(z.requires_grad)
y.requires_grad_(True)
a = x + y
print(a.requires_grad)

#If a tensor is a result of an operation in which at least one tensor has requires_grad = True, the output tensor will have requires_grad = True
# If no input tensor has requires grad = True, output will have requires_grad = False
# Changing the requires_grad attribute by using tensor.requires_grad_(True) function

#Creating custome datasets

x = np.random.normal(0.0,1.0,[100,10])
y = np.sum(np.sin(x), axis = 1)
print(x[7])
print(len(y))

class SinData(torch.utils.data.Dataset):

    def __init__(self):
        self.x = np.random.normal(0.0,1.0,[100,10])
        self.y = np.sum(np.sin(x), axis = 1)

    def __getitem__(self, index):
        data = torch.from_numpy(self.x[index])
        label = torch.from_numpy(np.array(self.y[index]))
        return data,label

    def __len__(self):
        return 100

data = SinData()

train_loader = torch.utils.data.DataLoader(dataset=data,batch_size=20,shuffle=True)

for d,l in train_loader:
    print("Data")
    print(d)
    print("Label")
    print(l)