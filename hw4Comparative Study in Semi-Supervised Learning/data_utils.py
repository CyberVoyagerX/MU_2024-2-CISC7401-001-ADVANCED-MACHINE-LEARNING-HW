import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import RandAugment 
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Dict, Union, Callable, Optional, Any
import matplotlib.pyplot as plt
from PIL import Image
DATA_ROOT = "./data"
NUM_CLASSES = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Standard/Weak Augmentation
transform_weak = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=4, padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
])

transform_labeled = transform_weak

# Strong Augmentation
transform_strong = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=4, padding_mode='reflect'),
    RandAugment(num_ops=2, magnitude=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
])

# Transform for the test set
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
])
base_transformer = transforms.Compose([
    transforms.ToTensor()
])

class CIFAR_SSL(Dataset):
    def __init__(self,
                 base_dataset: Dataset,
                 indices: List[int],
                 transform: Union[Callable, Tuple[Callable, ...]],
                 return_label: bool = True):
        super().__init__()
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.return_label = return_label
        
        self.is_multi_view = isinstance(transform, (list, tuple))
        
        self.data = [self.base_dataset.data[i] for i in indices]
        
        if return_label is True:
            self.labels = [self.base_dataset.targets[i] for i in indices]
        else:
            self.labels = None
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        '''
        Returns:
            Main differencies are in transform process
            1. laebled data -> weak transform images and laebl
            2. unlbaeled data
                -> 1. is_multi_view: For FixMatch
                -> 2. not is_multi_view: For MT, PL
        '''
        img_data = self.data[index]
        label = self.labels[index] if self.labels is not None else -1
        
        # Because self.data is a ndarray type data, so we should transform it to PIL
        # And in transform stage we can use transform(img_pil)
        img_pil = Image.fromarray(np.uint8(img_data))
        
        if self.is_multi_view:
            transformed_views = [tf(img_pil) for tf in self.transform]
            return tuple(transformed_views)
        else:
            transformed_img = self.transform(img_pil)
            if self.return_label:
                return transformed_img, label
            else:
                return transformed_img
        

def get_dataset(root: str = DATA_ROOT, transformer: transforms = base_transformer) -> Tuple[Dataset,Dataset]:
    """
    Get Dataset by torchvision
    args:
        root: cache dir
        transformer: basic methods of transformers
    
    return:
        Train data set and Test data set
    """
    Train_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=transformer,
        download=True,
    )
    Test_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        transform=transformer,
        download=True,
    )
    
    return Train_dataset, Test_dataset

def split_dataset(dataset: Dataset, num_per_class: int, num_classes: int = NUM_CLASSES, seed: int = 12) -> Tuple[Subset, Subset]:
    """
    Split Dataset into labeled and unlabeled subsets based on indices
    Args:
        dataset: the full dataset
        num_per_class: the number of samples in per class
        num_classes: the number of kinds in the full dataset
        seed: random seed
    Returns:
        indices of different dataset in original dataset (laebled_indices, unlabeled_indeces)
    """
    np.random.seed(seed)
    # label of CIFAR10
    targets = np.array(dataset.targets)
    labeled_indices = []
    unlabeled_indices = []
    
    # group indices by class
    class_indices = {}
    for idx, label in enumerate(targets):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Sample number of num_per_class in per class 
    for per_class in range(num_classes):
        # Get one class indices and shuffle it 
        indices = class_indices[per_class]
        np.random.shuffle(indices)
        
        labeled_indices.extend(indices[:num_per_class])
        unlabeled_indices.extend(indices[num_per_class:])
    
    # Finally we get labeld_indeces [[class1 labels] [class2 labels] ...] and unlabeld_indices
    return labeled_indices, unlabeled_indices


def get_ssl_dataloaders(
    n_labeled_per_class: int,
    batch_size_labeled: int,
    batch_size_unlabeled: int,
    num_workers: int = 4,
    root: str = DATA_ROOT,
    seed: int = 42,
    # Specify transforms directly for flexibility
    labeled_transform: Callable = transform_labeled,
    unlabeled_transform: Union[Callable, Tuple[Callable, ...]] = (transform_weak, transform_strong), # Default for FixMatch
    test_transform: Callable = transform_test
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''
    Return target data augmentation label_dataloader, unlabel_dataloader, test_dataloader
    Args:
        n_labeled_per_class: For split_dataset
        batch_size_labeled: For Dataloader load batch labeled data
        batch_size_unlabeled: For Dataloader load batch unlbaeled data
        num_workers: For Dataloader load data with multi line
        root: Data cache root
        seed: shuffle dataloader
        
        labeled_transform: augmentation method for labeled data
                            Normally, it is a weak method
        unlabeled_transform: augmentation method for unlabeled data
                                Normally, it is a strong method
        test_transformer: augmentation method for test data
                            Normally, it is just a norlization method
                            
    Returns: 
        All data loaders: (labeled_loader, unlabeled_loader, test_loader)
    '''
    print("\n--- Preparing DataLoaders ---")
    # Load the base dataset
    base_train_dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=None # Load as PIL
    )

    # Split the training dataset
    print(f"Splitting dataset with {n_labeled_per_class} labeled samples per class...")
    labeled_indices, unlabeled_indices = split_dataset(
        base_train_dataset, n_labeled_per_class, NUM_CLASSES, seed
    )

    # Create Labeled Dataset instance
    print("Creating Labeled Dataset...")
    labeled_dataset = CIFAR_SSL(
        base_dataset=base_train_dataset,
        indices=labeled_indices,
        transform=labeled_transform,
        return_label=True
    )

    # Create Unlabeled Dataset instance
    print("Creating Unlabeled Dataset...")
    unlabeled_dataset = CIFAR_SSL(
        base_dataset=base_train_dataset,
        indices=unlabeled_indices,
        transform=unlabeled_transform,
        return_label=False
    )

    # Create Test Dataset instance
    print("Creating Test Dataset...")
    test_dataset = torchvision.datasets.CIFAR10(
         root=root, train=False, download=True, transform=test_transform
    )


    # Create DataLoaders
    print("Creating DataLoaders...")
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size_labeled,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, # Improves data transfer speed to GPU
        drop_last=True,   # Ensure consistent batch sizes during training
        persistent_workers=True
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size_unlabeled,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_labeled + batch_size_unlabeled,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("DataLoaders created successfully.")
    print(f"Labeled Dataset size {len(labeled_indices)}")
    print(f"Unlabeled Dataset size {len(unlabeled_indices)}")
    return labeled_loader, unlabeled_loader, test_loader

def imshow_tensor(inp_tensor: torch.Tensor, title: Optional[str] = None, mean=CIFAR10_MEAN, std=CIFAR10_STD, ax=None):
    """
    Displays a PyTorch Tensor image after denormalizing it.

    Args:
        inp_tensor: Input image tensor (C x H x W).
        title: Optional title for the image.
        mean: Mean used for normalization.
        std: Standard deviation used for normalization.
        ax: Matplotlib axes object to plot on. If None, uses plt.imshow().
    """
    # Clone the tensor so we don't change the original
    image = inp_tensor.clone().cpu() # Move to CPU if necessary

    # Denormalize: Multiply by std and add mean
    # Ensure mean and std are tensors and reshape for broadcasting
    mean = torch.tensor(mean).view(-1, 1, 1) # Shape [C, 1, 1]
    std = torch.tensor(std).view(-1, 1, 1)   # Shape [C, 1, 1]
    image = image * std + mean

    # Convert tensor to numpy array and transpose dimensions
    # PyTorch: C x H x W -> NumPy: H x W x C
    np_image = image.numpy().transpose((1, 2, 0))

    # Clip values to [0, 1] range
    np_image = np.clip(np_image, 0, 1)

    # Display the image
    if ax:
        ax.imshow(np_image)
        if title:
            ax.set_title(title)
        ax.axis('off')
    else:
        plt.imshow(np_image)
        if title:
            plt.title(title)
        plt.axis('off')