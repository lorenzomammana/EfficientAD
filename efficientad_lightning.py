from pytorch_lightning import Trainer
from efficientad.datamodules import EfficientAdDataModule
from efficientad.modules.base import EfficientAd
from pytorch_lightning.loggers import MLFlowLogger
import dotenv
import os


if __name__ == "__main__":
    dotenv.load_dotenv()

    MVTEC_PATH = "/home/lmammana/shared/generic/mvtec"
    IMAGENET_PATH = "/home/lmammana/shared/generic/imagenet_images_564k"
    datamodule = EfficientAdDataModule(
        anomaly_data_path=MVTEC_PATH,
        imagenet_data_path=IMAGENET_PATH,
        category="bottle",
    )

    model = EfficientAd(
        teacher_pretrained_weights="./models/teacher_small.pth",
        model_size="small",
        out_channels=384,
        optimizer=None,
        lr_scheduler=None,
        lr_scheduler_interval="step",
        max_steps=70000,
    )

    logger = MLFlowLogger(experiment_name="EfficientAD-repo", tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    trainer = Trainer(
        logger=logger,
        devices=[1],
        accelerator="gpu",
        max_epochs=70000,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)
