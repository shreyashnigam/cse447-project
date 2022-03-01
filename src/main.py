import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

import model
import lightning_wrapper
import data_util
import text_dataset


if __name__ == "__main__":
    sequence_length = 64
    embed_dim = 192

    train_path = Path("data") / Path("cleanspanish.txt")
    # with open(train_path) as train_data:
    #     indexer = data_util.SymbolIndexer(train_data.read())
    #     print(indexer.size())
    #     print(indexer._known_symbol_to_index)
    train_path = "your/train/path"
    indexer = data_util.SymbolIndexer.russianandspanish()

    dataset = text_dataset.TextDataset(sequence_length, train_path, indexer=indexer)
    function = model.BasicModel(sequence_length, indexer, embed_dim)

    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=6, shuffle=True)
    checkpoint_callback = ModelCheckpoint(every_n_train_steps=1024)
    trainer = pl.Trainer(gpus=1, callbacks=[checkpoint_callback])

    trainer.fit(lightning_wrapper.LightningWrapper(function), loader)
