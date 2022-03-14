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


DEVICE = 'cpu'

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

        japanese_model = model.BasicModel(CONFIG_JAPANESE.sequence_length,
                                data_util.SymbolIndexer.japanese(), CONFIG_JAPANESE.embed_dim)
        japanese_model = lightning_wrapper.LightningWrapper.load_from_checkpoint(
            CONFIG_JAPANESE.chkpt_path, map_location=CONFIG_JAPANESE.device, f=japanese_model)
        
        norwegian_model = model.BasicModel(CONFIG_NORWEGIAN.sequence_length,
                                           data_util.SymbolIndexer.norwegian(), CONFIG_NORWEGIAN.embed_dim)
        norwegian_model = lightning_wrapper.LightningWrapper.load_from_checkpoint(
            CONFIG_NORWEGIAN.chkpt_path, map_location=CONFIG_NORWEGIAN.device, f=norwegian_model)

        chinese_model = model.BasicModel(CONFIG_CHINESE.sequence_length,
                                           data_util.SymbolIndexer.chinese(), CONFIG_CHINESE.embed_dim)
        chinese_model = lightning_wrapper.LightningWrapper.load_from_checkpoint(
            CONFIG_CHINESE.chkpt_path, map_location=CONFIG_CHINESE.device, f=chinese_model)

        hindi_model = model.BasicModel(CONFIG_HINDI.sequence_length,
                                           data_util.SymbolIndexer.hindi(), CONFIG_HINDI.embed_dim)
        hindi_model = lightning_wrapper.LightningWrapper.load_from_checkpoint(
            CONFIG_HINDI.chkpt_path, map_location=CONFIG_HINDI.device, f=hindi_model)

        french_model = model.BasicModel(CONFIG_FRENCH.sequence_length,
                                           data_util.SymbolIndexer.french(), CONFIG_FRENCH.embed_dim)
        french_model = lightning_wrapper.LightningWrapper.load_from_checkpoint(
            CONFIG_FRENCH.chkpt_path, map_location=CONFIG_FRENCH.device, f=french_model)


        self.my_models = {'en': english_model.f, 'es': spanish_model.f, 'ru': russian_model.f, 'ja': japanese_model.f, 'no': norwegian_model.f, 'zh': chinese_model.f, 'hi': hindi_model.f, 'fr': french_model.f}
        self.configs = {'en': CONFIG_ENGLISH, 'es': CONFIG_SPANISH, 'ru': CONFIG_RUSSIAN, 'ja': CONFIG_JAPANESE, 'no': CONFIG_NORWEGIAN, 'zh': CONFIG_CHINESE, 'hi': CONFIG_HINDI, 'fr': CONFIG_FRENCH}


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
        device=DEVICE,
        sequence_length=64,
        embed_dim=192,
    )

    CONFIG_SPANISH = MyModelConfig(
        chkpt_path="work/spanish.ckpt",
        dummy_prompt="Tomebamba también conocido como Tumipampa fue el centro administrativo del norte del Imperio inca, antes de la conquista Inca era el asentamiento cañari de Guapondelig. ",
        device=DEVICE,
        sequence_length=64,
        embed_dim=192,
    )

    CONFIG_RUSSIAN = MyModelConfig(
        chkpt_path="work/russian.ckpt",
        dummy_prompt="старейшее из существующих семейство с территории Южных Нидерландов, сначала баронское, затем княжеское. Упоминается на страницах источников с XII века. ",
        device=DEVICE,
        sequence_length=64,
        embed_dim=192,
    )

    CONFIG_JAPANESE = MyModelConfig(
        chkpt_path="work/japanese.ckpt",
        dummy_prompt="ヒトにおいて19遺伝子存在するアルデヒドデヒドロゲナーゼ遺伝子の1つであり、コードしているALDH2タンパク質はヒトの肝臓を中心に様々な組織、細胞においてエタノールの代謝産物であるアセトアルデヒドを含む反応性アルデヒドの酸化および無毒化に重要な働きをしている酵素である",
        device=DEVICE,
        sequence_length=64,
        embed_dim=192,
    )

    CONFIG_NORWEGIAN = MyModelConfig(
        chkpt_path="work/norwegian.ckpt",
        dummy_prompt="ar ikke noe sånn vondt ment det ei nlle skulle rette opp så du en feil så skulle n kke sant a det internt og for at det skulle være bra når det ga ja da syns vi var veldig e det er veldig e fornø",
        device=DEVICE,
        sequence_length=64,
        embed_dim=192,
    )

    CONFIG_CHINESE = MyModelConfig(
        chkpt_path="work/chinese.ckpt",
        dummy_prompt="是一种较弱的雌性甾体性激素，是三种主要的内源性雌激素之一，另外两种为雌二醇和雌三醇。雌酮等雌激素的生物合成从膽固醇开始，大部分由生殖腺分泌，少部分为脂肪組織对来自腎上腺的雄激素的转化。相对于雌二醇而言，雌酮和雌三醇的活性都很小。雌酮可转化为雌二醇，是主要的雌二醇代謝前体。 孩童+青春期 出生1-14 天：新生兒的雌酮水平在出生時非常高，但會在幾天內降至青春期前水平。 男性 #青春期開始",
        device=DEVICE,
        sequence_length=64,
        embed_dim=192,
    )

    CONFIG_HINDI = MyModelConfig(
        chkpt_path="work/hindi.ckpt",
        dummy_prompt="प्रतिशत तक क्षाराभ पाए जाते हैं जिनमें रिसरपिन प्रमुख हैं इसका गुण रूक्ष, रस में तिक्त, विपाक में कटु और इसका प्रभाव निद्राजनक होता है।",
        device=DEVICE,
        sequence_length=64,
        embed_dim=192,
    )

    CONFIG_FRENCH = MyModelConfig(
        chkpt_path="work/french.ckpt",
        dummy_prompt="Cette géométrie change significativement lors de la chélation : Les éthers couronnes peuvent être utilisés au laboratoire comme catalyseurs de transfert de phase, bien qu'il existe de tels catalyseurs moins chers et moins spécifiques. En présence de 18-C-6, le permanganate de potassium se dissout dans le benzène en donnant du benzène violet",
        device=DEVICE,
        sequence_length=64,
        embed_dim=192,
    )

    if args.mode == "train":
        if not os.path.isdir(args.work_dir):
            print("Making working directory {}".format(args.work_dir))
            os.makedirs(args.work_dir)
    elif args.mode == "test":
        set_languages(['en', 'es', 'ru', 'ja', 'no', 'zh', 'hi', 'fr'])
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
        set_languages(['en', 'es', 'ru', 'ja', 'no', 'zh', 'hi' ,'fr'])
        model = MyModel()
        user_prompt = ""
        while True:
            print(model.prediction_from_line(user_prompt, k=3))
            user_prompt += input(user_prompt)

    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))
