# from janome.tokenizer import Tokenizer
import re
import mojimoji
import string
import MeCab


import numpy as np
import random
import torch


import torchtext
from torchtext.vocab import Vectors


import sys
import pickle

sys.path.append("./../")


# Setup seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# j_t = Tokenizer()


class Make_DataLoader(object):
    def __init__(
        self,
        max_length: int = 256,
        lower=True,
        include_lengths=True,
        batch_first=True,
        init_token="<cls>",
        eos_token="<eos>",
        sequential=False,
        use_vocab=False,
    ):

        self.max_length = max_length
        self.sequential = sequential
        self.tokenize = self.tokenizer_with_preprocessing
        self.use_vocab = True
        self.lower = True
        self.include_lengths = True
        self.batch_first = True
        self.fix_length = max_length
        self.init_token = "<cls>"
        self.eos_token = "<eos>"
        self.use_vocab = False

    def tokenizer_mecab(self, text):
        m_t = MeCab.Tagger(
            "-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
        text = m_t.parse(text)  # これでスペースで単語が区切られる
        ret = text.strip().split()  # スペース部分で区切ったリストに変換
        return ret

    def preprocessing_text(self, text):

        # 半角・全角の統一
        text = mojimoji.han_to_zen(text)
        # 改行、半角スペース、全角スペースを削除
        text = re.sub("\r", "", text)
        text = re.sub("\n", "", text)
        text = re.sub("　", "", text)
        text = re.sub(" ", "", text)
        # 数字文字の一律「0」化
        text = re.sub(r"[0-9 ０-９]+", "0", text)  # 数字

        # カンマ、ピリオド以外の記号をスペースに置換
        for p in string.punctuation:
            if (p == ".") or (p == ","):
                continue
            else:
                text = text.replace(p, " ")

        return text

    # 前処理関数
    def tokenizer_with_preprocessing(self, text):
        text = self.preprocessing_text(text)  # 前処理の正規化
        ret = self.tokenizer_mecab(text)  # Janomeの単語分割

        return ret

    def make_ds_and_TEXT(
        self,
        train="train.tsv",
        validation="validation.tsv",
        test="all_sentence_make_for_vocab.tsv",
    ):

        TEXT = torchtext.data.Field(
            sequential=True,
            tokenize=self.tokenizer_with_preprocessing,
            use_vocab=True,
            lower=True,
            include_lengths=True,
            batch_first=True,
            fix_length=self.max_length,
            init_token="<cls>",
            eos_token="<eos>",
        )
        LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
        # init_token：全部の文章で、文頭に入れておく単語
        # eos_token：全部の文章で、文末に入れておく単語

        train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
            path="./data/",
            train=train,
            validation=validation,
            test=test,
            format="tsv",
            fields=[("Text", TEXT), ("Label", LABEL)],
        )
        datasets_dict = {"train": train_ds, "val": val_ds, "test": test_ds}

        # torchtextで日本語ベクトルとして日本語学習済みモデルを読み込む
        japanese_fastText_vectors = Vectors(name="./data/model.vec")

        # ベクトル化したバージョンのボキャブラリーを作成します
        TEXT.build_vocab(
            test_ds, vectors=japanese_fastText_vectors, min_freq=3)

        self.pickle_dump(TEXT=TEXT, path="./data/TEXT.pkl")

        return datasets_dict, TEXT

    # DataLoaderを作成
    def make_dl(self, datasets_dict):

        train_ds, val_ds, test_ds = (
            datasets_dict["train"],
            datasets_dict["val"],
            datasets_dict["test"],
        )

        train_dl = torchtext.data.Iterator(train_ds, batch_size=8, train=True)

        val_dl = torchtext.data.Iterator(
            val_ds, batch_size=8, train=False, sort=False)

        test_dl = torchtext.data.Iterator(
            test_ds, batch_size=8, train=False, sort=False
        )

        dataloaders_dict = {"train": train_dl, "val": val_dl, "test": test_dl}

        return dataloaders_dict

    def pickle_dump(self, TEXT, path):
        with open(path, "wb") as f:
            pickle.dump(TEXT, f)

    def pickle_load_TEXT(self, path=None):
        if path is None:
            path = "./data/TEXT.pkl"
        with open(path, "rb") as f:
            TEXT = pickle.load(f)
        return TEXT

    def create_tensor(self, text, max_length, TEXT_pkl_path="./data/TEXT.pkl"):
        # 入力文章をTorch Teonsor型にのINDEXデータに変換
        TEXT = self.pickle_load_TEXT(path=TEXT_pkl_path)
        token_ids = torch.ones((max_length)).to(torch.int64)
        ids_list = list(map(lambda x: TEXT.vocab.stoi[x], text))
        # print(ids_list)
        for i, index in enumerate(ids_list):
            token_ids[i] = index
        return token_ids
