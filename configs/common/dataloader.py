from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from detectron2.config import LazyCall as L
from torch.utils.data.distributed import DistributedSampler

from data import ImageFile, DataGenerator

# Dataloader
train_dataset = DataGenerator(
    data=ImageFile(
        images_dir="data/train/images",
        alphas_dir="data/train/alphas",
        trimaps_dir="data/train/trimaps",
    )
)

dataloader = OmegaConf.create()
dataloader.train = L(DataLoader)(
    dataset=train_dataset,
    batch_size=15,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=L(DistributedSampler)(
        dataset=train_dataset,
    ),
    drop_last=True,
)
