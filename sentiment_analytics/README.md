#  ネガポジ分析

## Requirements

- Python3

- Pytorch  

## Installation

```
$ pip install -r requirements.txt  
```

[drive](https://drive.google.com/drive/u/0/folders/1UCLlwcSdeiA9Fxc_ghoXIHFuCK6GXDd1)からtorchtextで日本語ベクトルとして学習済させたモデル(model.vec)をダウンロードして

`sentiment-analytics/negapoji_analytics/src/data/` に置いてください．


## Usage

詳しくは`negapoji_analytics/src/`下の Transformer.ipynb を参考にしてください．

### 推論

モジュールのimport及びclassの初期化。

テキストを推論したいテキストをlist型にして`predict_np()`の引数に渡す。

```python
from modules.sentiment_analytics import Analytics

negapoji=Analytics()
text_list=['ああ', 'いい']

negapoji.predict_np(text_list)
```

#### 結果

~~~
pe.shape= torch.Size([256, 300])

ああ
Negative
Negative度 : 1.1369767785072324

いい
Positive
Positive度 : 2.072095811367035
{0: ['Negative', 1.1369767785072324], 1: ['Positive', 2.072095811367035]}
~~~

`predict_np()` は `text_list`のindex番号をkeyにとり推論結果(Negative または Positive)と、そのスコアをlist型で返す。



## 学習済みモデル

#### データセット 

TIS株式会社さんが無償公開している「[chABSA-dataset](https://www.tis.co.jp/news/2018/tis_news/20180410_1.html)」という上場企業の有価証券報告書(2016年度)をベースに作成された感情分析用のデータセットを用いて学習させました。

#### モデル

このリポジトリの感情分析ではTransofomerを使って実装しました。

実装するにあたり、「[つくりながら学ぶ! PyTorchによる発展ディープラーニング](https://www.amazon.co.jp/dp/4839970254/)」という書籍及びその[サポートリポジトリ](https://github.com/YutaroOgawa/pytorch_advanced)と[sinyさんのリポジトリ](https://github.com/sinjorjob/django-transformer)を参考にさせていただきました。














