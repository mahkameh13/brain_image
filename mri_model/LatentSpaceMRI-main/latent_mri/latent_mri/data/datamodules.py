import lightning as L
from torch.utils.data import DataLoader, Subset, Sampler


class MRIDataModule(L.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.train_set = None
        self.valid_set = None
        self.test_set = None

        self.data_split = None

    def create_datasets(self, partial_dataset, data_split, transforms):
        self.data_split = data_split

        indices_train, _ = data_split['train']
        indices_valid, _ = data_split['valid']
        indices_test, _ = data_split['test']

        transforms_train = transforms['train']
        transforms_valid = transforms['valid']
        transforms_test = transforms['test']

        self.train_set = Subset(partial_dataset(transforms=transforms_train),
                                indices=indices_train)
        self.valid_set = Subset(partial_dataset(transforms=transforms_valid),
                                indices=indices_valid)
        self.test_set = Subset(partial_dataset(transforms=transforms_test),
                               indices=indices_test)

    def print(self):
        print(f'Num. Subjects in Train: {len(self.train_set)}')
        print(f'Num. Subjects in Valid: {len(self.valid_set)}')
        print(f'Num. Subjects in Test: {len(self.test_set)}')

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          **self.params['train'])

    def val_dataloader(self):
        return DataLoader(self.valid_set,
                          **self.params['valid'])

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          **self.params['test'])

    def predict_dataloader(self):
        return self.test_dataloader()


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices) -> None:
        self.indices = indices

    def __iter__(self):
        for index in self.indices:
            yield index

    def __len__(self) -> int:
        return len(self.indices)
