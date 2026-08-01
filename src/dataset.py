from .features import get_transform
from torchvision import datasets
from torch.utils.data import DataLoader

def get_dataloaders(data_dir,batch_size=50):
    transforms = get_transform()
    
    train_data = datasets.FashionMNIST(data_dir,True,transforms,download=True)
    test_data = datasets.FashionMNIST(data_dir,False,transforms,download=True)
    
    train_loader = DataLoader(train_data,batch_size,True)
    test_loader = DataLoader(test_data,batch_size,True)
    
    return train_loader , test_loader