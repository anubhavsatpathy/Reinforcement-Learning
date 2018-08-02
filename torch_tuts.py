import torch
import numpy as np

#Creating an uninitialized tensor of a given shape
x = torch.empty(size = [4,4],dtype = torch.int32)
print(x)

#Sampling a tensor from a uniform distribution in R[0,1)
x = torch.rand(size = [4,4], dtype = torch.float32)
print(x)

#Creating a 0/1 tensor of a given shape and type
x = torch.zeros(size = [4,4], dtype = torch.int32)
print(x)
x = torch.ones(size = [4,4], dtype = torch.long)
print(x)

#Zero like and one like and rand like
x = torch.randn_like(x,dtype = torch.float32)
print(x)
x = x.new_ones(size = [5,5]) # Takes the type and the device of the object tensor if not explicitly provided
print(x)

#Get the shape of a tensor
print(x.size())

#Adding two tensors
y = torch.randn_like(x)
print(y)
print(x+y)
z = torch.add(x,y)
print(z)
z1 = torch.empty(size = [5,5])
torch.add(x,y,out = z1) # Provide output tensor as argu,emt
print(z1)
y.add_(x) # inplace addition
print(y)

#Indexing and Reshaping
p = z[:,1]
print(p)
q = z.view(25)
print(q)

# Transfering to and from numpy array

a = torch.ones(size = [4], dtype = torch.long)
print(a)
b = a.numpy()
a = a.add_(1)
print(a)
print(b)

c = np.ones(4)
b = torch.from_numpy(c)
np.add(c,1,out = c)
print(b)

#Taking gradients

x = torch.ones(size = [4,4], requires_grad = True)
y = torch.empty(size = [4,4])
for i in range(4):
    y[i,:] = (i+1)*x[i,:] + 4
y.backward(torch.ones([4,4]))
print(x.grad)






