from .common.train import train
from .common.model import model
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.dataloader import dataloader


train.max_iter = int(43100 / 16 / 1 * 10)
train.checkpointer.period = int(43100 / 16 / 1 * 1)
# train.max_iter = int(120 / 16 / 2 * 100)
# train.checkpointer.period = int(120 / 16 / 2 * 10)

optimizer.lr = 5e-4
lr_multiplier.scheduler.values = [1.0, 0.1, 0.05]
lr_multiplier.scheduler.milestones = [
    int(43100 / 16 / 1 * 3),
    int(43100 / 16 / 1 * 9),
    # int(120 / 16 / 2 * 30),
    # int(120 / 16 / 2 * 90),
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 5 / train.max_iter

# train.init_checkpoint = './pretrained/dino_vit_s_fna.pth'
# train.output_dir = './output_of_train/ViTMatte_S_100ep'
train.init_checkpoint = "checkpoints/ViTMatte_S_Com.pth"
train.output_dir = "output_of_train/ViTMatte_S_100ep"

dataloader.train.batch_size = 16
# dataloader.train.num_workers = 2
dataloader.train.num_workers = 1
