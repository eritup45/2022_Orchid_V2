from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as transforms
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

# IMAGE_SIZE = (256, 256)
# def images_transforms(phase):
#     if phase == 'training':
#         data_transformation =transforms.Compose([
#             transforms.Resize(IMAGE_SIZE),
#             transforms.RandomEqualize(10),
#             transforms.RandomRotation(degrees=(-25,20)),
#             # transforms.CenterCrop(64),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
#         ])
#     else:
#         data_transformation=transforms.Compose([
#             transforms.Resize(IMAGE_SIZE),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
#         ])
#     return data_transformation

def my_get_loader(args):
    train_transform=transforms.Compose([
                                transforms.Resize((600, 600), Image.BILINEAR),
                                transforms.RandomCrop((448, 448)),
                                
                                # transforms.Resize((500, 500), Image.BILINEAR),
                                # transforms.RandomCrop((350, 350)),
                                
                                # transforms.CenterCrop((448, 448)),
                                transforms.RandomHorizontalFlip(),
                                # transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.05),
                                # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5))], p=0.1),
                                transforms.RandomApply([transforms.RandomAffine(degrees=15)], p=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform=transforms.Compose([
                                transforms.Resize((600, 600), Image.BILINEAR),
                                transforms.CenterCrop((448, 448)),

                                # transforms.Resize((500, 500), Image.BILINEAR),
                                # transforms.CenterCrop((350, 350)),

                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trainset = TrainOrchidDataset(transform=train_transform)
    testset = ValOrchidDataset(transform=test_transform)

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader



class TrainOrchidDataset(Dataset):
    def __init__(self, root_dir='./dataset/', csv_file='dataset/train_label.csv', transform=None):
        self.x = []
        self.y = []
        self.transform = transform
        sample_submission = pd.read_csv(csv_file)
        for idx, filename in enumerate(sample_submission['filename']):
            # print("filename: ", filename,"Label: ",sample_submission['category'][idx])
            filename = os.path.join(root_dir, filename)
            self.x.append(filename)
            self.y.append(sample_submission['category'][idx])
        self.num_classes = max(self.y)+1
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.y[index]
    
class ValOrchidDataset(Dataset):
    def __init__(self, root_dir='./dataset/', csv_file='dataset/val_label.csv', transform=None):
        self.x = []
        self.y = []
        self.transform = transform
        sample_submission = pd.read_csv(csv_file)
        for idx, filename in enumerate(sample_submission['filename']):
            # print("filename: ", filename,"Label: ",sample_submission['category'][idx])
            filename = os.path.join(root_dir, filename)
            self.x.append(filename)
            self.y.append(sample_submission['category'][idx])
        self.num_classes = max(self.y)+1
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.y[index]
    

if __name__ == '__main__':
    d = TrainOrchidDataset('./dataset/images/', 'dataset/train_label.csv')
    print(len(d))
    print(d[10])
    pass