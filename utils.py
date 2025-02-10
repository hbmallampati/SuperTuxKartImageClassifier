from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use your solution (or the master solution) to HW1
        """
        import csv
        from os import path
        
        
        self.data = []
        image_to_tensor = transforms.ToTensor()
        with open(path.join(dataset_path, 'labels.csv'), newline='') as f:
            dataset_reader = csv.reader(f)
            
            for file_name, label, _ in dataset_reader:
                if label in LABEL_NAMES:
                    image = Image.open(path.join(dataset_path, file_name))
                    image_label = LABEL_NAMES.index(label)
                    self.data.append((image_to_tensor(image), image_label))

    def __len__(self):
        """
        Your code here
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        """
        return self.data[idx]


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
