import lightning as L
from torch.utils.data import DataLoader

from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData, Shapes3dData, SpritesData
from disent.dataset.transform import ToImgTensorF32
from disent.frameworks.vae import BetaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64
from disent.model.ae import EncoderConv64
from disent.util import is_test_run  # you can ignore and remove this

# prepare the data
data = Shapes3dData(prepare=False)
# data = SpritesData(prepare=False)

dataset = DisentDataset(data, transform=ToImgTensorF32())
dataloader = DataLoader(dataset=dataset, batch_size=512, shuffle=True, num_workers=20)

# create the pytorch lightning system
module: L.LightningModule = BetaVae(
    model=AutoEncoder(
        encoder=EncoderConv64(x_shape=data.x_shape, z_size=16, z_multiplier=2),
        decoder=DecoderConv64(x_shape=data.x_shape, z_size=16),
    ),
    cfg=BetaVae.cfg(optimizer="adam", optimizer_kwargs=dict(lr=1e-3), loss_reduction="mean_sum", beta=4),
)

# train the model
trainer = L.Trainer(logger=True, fast_dev_run=is_test_run(), max_epochs=10, 
                    enable_checkpointing=True, default_root_dir='checkpoints/shapes3d-betaVAE')
trainer.fit(module, dataloader)




