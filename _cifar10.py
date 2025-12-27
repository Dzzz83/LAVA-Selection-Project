import torch
import torchvision
from PIL import Image

class cifar10_dataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )

    def __getitem__(self, idx):
        # Get raw HWC NumPy image + label from CIFAR10
        image, label = self.data[idx], self.targets[idx]

        # Convert NumPy → PIL (CRITICAL)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Apply transform (ToTensor → CHW happens here)
        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx # change order
if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = cifar10_dataset(root='./data',
                               train=True,
                               download=True,
                               transform=transform)
    trainloader = iter(trainset)    
    data, label = next(trainloader)
    print(data.shape)   # MUST be [3, 32, 32]
    print(type(data))   # torch.Tensor
    print(label)  