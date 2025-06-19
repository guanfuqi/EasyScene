import torch

#labels = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 1, 2], [1, 3, 3], [2, 2, 2], [2, 3, 3], [3, 3, 4], [4, 0, 0] ])


def find_object(label):
    
    labels = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 1, 2], [1, 3, 3], [2, 2, 2], [2, 3, 3], [3, 3, 4], [4, 0, 0] ])
    in_sequence = False
    start = 0
    end = 0
    labels_1d = labels[: , 0:1].clone()
    
    for idx in range(labels_1d.shape[0]):
        print(labels_1d[idx])
        if labels_1d[idx]==label and not in_sequence:
            start = idx
            in_sequence = True
        if labels_1d[idx]!=label and in_sequence:
            end = idx
            in_sequence = False
            break
    if in_sequence and start is not None:
        end = len(labels_1d)
    print(label, start, end)
    return start, end
    
label_num = 9
obj_stats = {}
environment_label = 4

for label in range(label_num):
    if label == environment_label:
        continue
    start, end = find_object(label)
    obj_stats[label] = end-start
print(obj_stats)
obj_stats = sorted(obj_stats.items(), key=lambda item:item[1], reverse=True)
print(obj_stats)

'''
vertices = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 100], [5, 5, 5], [6, 6, 6], [7, 7, 7] ])
labels = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 1, 2], [1, 3, 3], [1, 2, 2], [1, 3, 3], [1, 3, 4] ])
label = 1

in_sequence = False
start = 0
end = 0
labels_1d = labels[: , 0:1]
print(labels_1d)


for idx in range(len(labels_1d)):
    print(labels_1d[idx])
    if labels_1d[idx]==label and not in_sequence:
        start = idx
        in_sequence = True
    if labels_1d[idx]!=label and in_sequence:
        end = idx-1
        in_sequence = False
        break
print(start, end)
if in_sequence and start is not None:
    end = len(labels_1d)-1
obj_vertices = vertices[start:end+1, :]

center = obj_vertices.mean(dim=0, dtype=torch.float64)
print(obj_vertices)
print(center)
'''