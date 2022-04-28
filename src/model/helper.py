from pytorch_lightning.callbacks import ModelCheckpoint

__author__ = "Maryam Najafi🥰"
__organization__ = "Religious ChatBot"
__license__ = "Public Domain"
__version__ = "1.1.0"
__email__ = "Maryam_Najafi73@yahoo.com"
__status__ = "Production"
__date__ = "07/27/2021"


def build_checkpoint_callback(save_top_k, filename="QTag-{epoch:02d}-{val_loss:.2f}",
                              monitor="val_loss"):
    # saves a file like: input/QTag-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,  # monitored quantity
        filename=filename,
        save_top_k=save_top_k,  # save the top k models
        mode="min",  # mode of the monitored quantity for optimization
    )
    return checkpoint_callback
