import torch
import numpy

a = torch.tensor(
    [[[11, 11, 11], [12, 12, 12], [13, 13, 13]], 
    [[21, 21, 21], [22, 22, 22], [23, 23, 23]]]
    )
b = torch.tensor(
    [[[1, 0, 0], [2, 0, 0], [3, 0, 0]], 
    [[4, 0, 0], [5, 0, 0], [6, 0, 0]]]
    )
print(a)
print(b)
c = torch.stack((a, b), dim=0)
print(c)
c = torch.stack((a, b), dim=1)
print(c)
c = torch.stack((a, b), dim=2)
print(c)
c = torch.stack((a, b), dim=3)
print(c)

ideal = torch.tensor(
    [[[11+100j, 11, 11], [12+100j, 12, 12], [12+100j, 13, 13]], 
    [[12+100j, 21, 21], [12+100j, 22, 22], [12+100j, 23, 23]]]
)
print(ideal)

test1 = torch.tensor(
    [[[11, 11, 11], [12, 12, 12], [12, 13, 13]], 
    [[12, 21, 21], [12, 22, 22], [12, 23, 23]]]
)
test2 = torch.tensor(
    [[[100j, 0, 0], [100j, 0, 0], [100j, 0, 0]], 
    [[100j, 0, 0], [100j, 0, 0], [100j, 0, 0]]]
)

test = test1 +test2
print(test)

test = test.numpy()
print(test)
print(numpy.real(test))
print(numpy.imag(test))