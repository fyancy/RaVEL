from torch.utils.data import Dataset, DataLoader


class InfiniteDataset(Dataset):
    r"""Dataset wrapping tensors.

        Each sample will be retrieved by indexing tensors along the first dimension.

        Args:
            *tensors (Tensor): tensors that have the same size of the first dimension.
        """
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.num_data = tensors[0].size(0)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index % self.num_data] for tensor in self.tensors)

    def __len__(self):
        return self.num_data


class InfiniteDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         pin_memory=pin_memory, collate_fn=None, drop_last=drop_last)

    def __iter__(self):
        return self.iter_function()

    def iter_function(self):
        while True:
            for batch in super().__iter__():
                yield batch
