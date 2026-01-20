import torch
tensor = torch.load('data/tensors/dataset_criticidade_janela180d.pt', weights_only=False)
print('Shape:', tensor.shape)
print('Type:', type(tensor))
