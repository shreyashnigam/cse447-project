#!/usr/bin/env python
import os
import string
from matplotlib import interactive
import torch
import random
from dataclasses import dataclass
from typing import List
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import data_util
import model
import lightning_wrapper
from langid import set_languages, classify


@dataclass
class MyModelConfig:
    dummy_prompt: str
    device: str
    chkpt_path: str
    sequence_length: int
    embed_dim: int

    def __post__init__(self):
        assert len(self.dummy_prompt) >= self.sequence_length


class MyModel:
    def __init__(self):
        english_model = model.BasicModel(CONFIG_ENGLISH.sequence_length,
                                data_util.SymbolIndexer.english(), CONFIG_ENGLISH.embed_dim)
        english_model = lightning_wrapper.LightningWrapper.load_from_checkpoint(
            CONFIG_ENGLISH.chkpt_path, map_location=CONFIG_ENGLISH.device, f=english_model)

        spanish_model = model.BasicModel(CONFIG_SPANISH.sequence_length,
                                data_util.SymbolIndexer.spanish(), CONFIG_SPANISH.embed_dim)
        spanish_model = lightning_wrapper.LightningWrapper.load_from_checkpoint(
            CONFIG_SPANISH.chkpt_path, map_location=CONFIG_SPANISH.device, f=spanish_model)

        russian_model = model.BasicModel(CONFIG_RUSSIAN.sequence_length,
                                data_util.SymbolIndexer.russian(), CONFIG_RUSSIAN.embed_dim)
        russian_model = lightning_wrapper.LightningWrapper.load_from_checkpoint(
            CONFIG_RUSSIAN.chkpt_path, map_location=CONFIG_RUSSIAN.device, f=russian_model)


        self.my_models = {'en': english_model.f, 'es': spanish_model.f, 'ru': russian_model.f}
        self.configs = {'en': CONFIG_ENGLISH, 'es': CONFIG_SPANISH, 'ru': CONFIG_RUSSIAN}
        # set_languages(['en', 'es', 'ru'])


    @classmethod
    def load_test_data(cls, fname):
        with open(fname) as f:
            data = [line[:-1].lower() for line in f]
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, "wt") as f:
            for p in preds:
                f.write(f"{p}\n")

    def prediction_from_line(self, line: str, k: int) -> str:
        lang_class = classify(line)   # returns a tuple of language id and -log prob
        config = self.configs[lang_class[0]]
        if len(line) >= config.sequence_length:
            line = line[-config.sequence_length:]
        else:    
            line = config.dummy_prompt[-(
                config.sequence_length - len(line)):] + line
        
        model = self.my_models[lang_class[0]]
        x = torch.ByteTensor([model.embed.indexer.to_index(symbol)
                              for symbol in line]).unsqueeze(0)
        y_pred = model(x).squeeze(0)[-1]
        result = model.embed.interpret(y_pred, k=k+1)
        result = [c for c in result if c is not None][:k]
        return "" .join(result)

    def run_pred(self, data: List[str]):
        # your code here
        preds: List[str] = []
        for line in data:
            preds.append(self.prediction_from_line(line, k=3))
        return preds


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=(
        "train", "test", "interactive"), help="what to run")
    parser.add_argument("--work_dir", help="where to save", default="work")
    parser.add_argument("--test_data", help="path to test data",
                        default="example/input.txt")
    parser.add_argument(
        "--test_output", help="path to write test predictions", default="pred.txt")
    args = parser.parse_args()

    CONFIG_ENGLISH = MyModelConfig(
        chkpt_path="work/english.ckpt",
        dummy_prompt="in other words, living an eternity of just about anything is now more terrifying to me than death. ",
        device="cpu",
        sequence_length=64,
        embed_dim=192,
    )

    CONFIG_SPANISH = MyModelConfig(
        chkpt_path="work/spanish.ckpt",
        dummy_prompt="Tomebamba también conocido como Tumipampa fue el centro administrativo del norte del Imperio inca, antes de la conquista Inca era el asentamiento cañari de Guapondelig. ",
        device="cpu",
        sequence_length=64,
        embed_dim=192,
    )

    CONFIG_RUSSIAN= MyModelConfig(
        chkpt_path="work/russian.ckpt",
        dummy_prompt="старейшее из существующих семейство с территории Южных Нидерландов, сначала баронское, затем княжеское. Упоминается на страницах источников с XII века. ",
        device="cpu",
        sequence_length=64,
        embed_dim=192,
    )

    if args.mode == "train":
        if not os.path.isdir(args.work_dir):
            print("Making working directory {}".format(args.work_dir))
            os.makedirs(args.work_dir)
    elif args.mode == "test":
        set_languages(['en', 'es', 'ru'])
        model = MyModel()
        print("Loading test data from {}".format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print("Making predictions")
        pred = model.run_pred(test_data)
        print("Writing predictions to {}".format(args.test_output))
        assert len(pred) == len(test_data), "Expected {} predictions but got {}".format(
            len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    elif args.mode == "interactive":
        set_languages(['en', 'es', 'ru'])
        model = MyModel()
        user_prompt = ""
        while True:
            print(model.prediction_from_line(user_prompt, k=3))
            user_prompt += input(user_prompt)

    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))
