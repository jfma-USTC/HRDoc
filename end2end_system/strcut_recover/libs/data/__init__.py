import torch
from .dataset import valid_collate_func, PickleLoader

def create_valid_dataloader(ly_vocab, re_vocab, pickle_path, batch_size, num_workers):
    dataset = PickleLoader(pickle_path, ly_vocab, re_vocab, mode='test')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=valid_collate_func,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    return dataloader

if __name__ == "__main__":
    pass