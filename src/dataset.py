from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def create_dataloader(data_dir, transform, batch_size, shuffle=True):
    dataset = ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
