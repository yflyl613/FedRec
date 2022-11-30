from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, user_id, user_data):
        super().__init__()
        self.uid = user_id
        self.user_data = user_data

    def __getitem__(self, idx):
        return (self.uid, self.user_data[idx])

    def __len__(self):
        return len(self.user_data)


class ContrastiveDataset(Dataset):
    def __init__(self, training_data, label):
        super().__init__()
        self.training_data = training_data
        self.label = label

    def __getitem__(self, idx):
        return (self.training_data[idx], self.label[idx])

    def __len__(self):
        return len(self.training_data)


class TestDataset(Dataset):
    def __init__(self, test_data):
        super().__init__()
        self.uid = test_data['uid']
        self.mask = test_data['mask']
        self.label = test_data['label']

    def __getitem__(self, idx):
        return self.uid[idx], self.mask[idx], self.label[idx]

    def __len__(self):
        return len(self.uid)
