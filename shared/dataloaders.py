import torchvision.transforms as transforms
import torchvision
import torch


def get_fashion_mnist_loader(batch_size=128, img_size=32):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),                
        transforms.Normalize((0.5,), (0.5,)) # Scales [0, 1] -> [-1, 1]
    ])

    trainset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )

    
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2  
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    return train_loader, test_loader