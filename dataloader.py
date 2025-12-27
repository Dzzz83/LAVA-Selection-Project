import torch 
from torch.utils.data import DataLoader, SubsetRandomSampler

def create_dataloader(
    train_dataset,        # Now accepts the OBJECT, not the name
    val_dataset,
    batch_size=512,
    num_workers=2,
    select_indices=None   # If None -> Baseline. If list -> Selection.
):
    """
    Wraps existing dataset objects into DataLoaders.
    Does NOT re-initialize datasets, preventing Index Mismatch bugs.
    """

    # 1. Create Sampler (if selection is active)
    sampler = None
    if select_indices is not None:
        # We use the indices calculated in the main script
        sampler = SubsetRandomSampler(select_indices)
    
    # 2. Create Train Loader
    # If sampler is present, shuffle must be False (sampler handles shuffling)
    # If sampler is None (Baseline), we set shuffle=True
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=(sampler is None) 
    )

    # 3. Create Val Loader
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    return train_loader, val_loader