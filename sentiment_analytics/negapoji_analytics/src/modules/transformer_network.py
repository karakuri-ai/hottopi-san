import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import torch.optim as optim

import matplotlib.pyplot as plt
import time


# Setup seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class Embedder(nn.Module):
    """idで示されている単語をベクトルに変換"""

    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True
        )
        # freeze=Trueによりバックプロパゲーションで更新されず変化しなくなる。

    def forward(self, x):
        x_vec = self.embeddings(x)

        return x_vec


class PositionalEncoder(nn.Module):
    """入力された単語の位置を示すベクトル情報を付加する"""

    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()

        self.d_model = d_model  # 単語ベクトルの次元数

        # 単語の順番（pos）と埋め込みベクトルの次元の位置（i）によって一意に定まる値の表をpeとして作成
        pe = torch.zeros(max_seq_len, d_model)

        # GPUが使える場合はGPUへ送る、ここでは省略。実際に学習時には使用する
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * (i + 1)) / d_model)))

        # 表peの先頭に、ミニバッチ次元となる次元を足す
        self.pe = pe.unsqueeze(0)
        print("pe.shape=", pe.shape)

        # 勾配を計算しないようにする
        self.pe.requires_grad = False

    def forward(self, x):

        # 入力xとPositonal Encodingを足し算する
        # xがpeよりも小さいので、大きくする
        ret = math.sqrt(self.d_model) * x + self.pe
        return ret


class Attention(nn.Module):
    """Transformerは本当はマルチヘッドAttentionですが、
    分かりやすさを優先しシングルAttentionで実装します"""

    def __init__(self, d_model=300):
        super().__init__()

        # SAGANでは1dConvを使用したが、今回は全結合層で特徴量を変換する
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        # 出力時に使用する全結合層
        self.out = nn.Linear(d_model, d_model)

        # Attentionの大きさ調整の変数
        self.d_k = d_model

    def forward(self, q, k, v, mask):
        # 全結合層で特徴量を変換
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        # Attentionの値を計算する
        # 各値を足し算すると大きくなりすぎるので、root(d_k)で割って調整
        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        # ここでmaskを計算
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)

        # softmaxで規格化をする
        normlized_weights = F.softmax(weights, dim=-1)

        # AttentionをValueとかけ算
        output = torch.matmul(normlized_weights, v)

        # 全結合層で特徴量を変換
        output = self.out(output)

        return output, normlized_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        """Attention層から出力を単純に全結合層2つで特徴量を変換するだけのユニットです"""
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)  # torch.Size([24, 256, 1024])
        self.dropout = nn.Dropout(dropout)  # torch.Size([24, 256, 1024])
        self.linear_2 = nn.Linear(d_ff, d_model)  # torch.Size([24, 256,300])

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # LayerNormalization層
        # https://pytorch.org/docs/stable/nn.html?highlight=layernorm
        self.norm_1 = nn.LayerNorm(d_model)  # 平均0　標準偏差1に正規化
        self.norm_2 = nn.LayerNorm(d_model)  # 平均0　標準偏差1に正規化

        # Attention層
        self.attn = Attention(d_model)

        # Attentionのあとの全結合層2つ
        self.ff = FeedForward(d_model)

        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 正規化とAttention
        x_normlized = self.norm_1(x)

        output, normlized_weights = self.attn(
            x_normlized, x_normlized, x_normlized, mask
        )

        x2 = x + self.dropout_1(output)

        # 正規化と全結合層
        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))

        return output, normlized_weights


class ClassificationHead(nn.Module):
    """Transformer_Blockの出力を使用し、最後にクラス分類させる"""

    def __init__(self, d_model=300, output_dim=2, learning_rate=2e-5):
        super().__init__()

        self.learning_rate = learning_rate

        # 全結合層
        self.linear = nn.Linear(d_model, output_dim)  # output_dimはポジ・ネガの2つ

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        x0 = x[:, 0, :]  # 各ミニバッチの各文の先頭の単語の特徴量（300次元）を取り出す
        out = self.linear(x0)

        return out


class TransformerClassification(nn.Module):
    """Transformerでクラス分類させる"""

    def __init__(
        self, text_embedding_vectors, d_model=300, max_seq_len=256, output_dim=2
    ):
        super().__init__()

        # モデル構築
        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.net3_1 = TransformerBlock(d_model=d_model)
        self.net3_2 = TransformerBlock(d_model=d_model)
        self.net4 = ClassificationHead(output_dim=output_dim, d_model=d_model)

    def forward(self, x, mask):
        x1 = self.net1(x)  # 単語をベクトルに
        x2 = self.net2(x1)  # Positon情報を足し算
        x3_1, normlized_weights_1 = self.net3_1(
            x2, mask)  # Self-Attentionで特徴量を変換
        x3_2, normlized_weights_2 = self.net3_2(
            x3_1, mask)  # Self-Attentionで特徴量を変換
        x4 = self.net4(x3_2)  # 最終出力の0単語目を使用して、分類0-1のスカラーを出力
        return x4, normlized_weights_1, normlized_weights_2


##################################
class Construct_Network(nn.Module):
    def __init__(
        self, TEXT, learning_rate=2e-5,
    ):
        super().__init__()

        self.TEXT = TEXT
        self.learning_rate = learning_rate
        self.net = TransformerClassification(
            text_embedding_vectors=self.TEXT.vocab.vectors,
            d_model=300,
            max_seq_len=256,
            output_dim=2,
        )
        self.criterion = None
        self.optimizer = None

    # ネットワークの初期化を定義

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            # Liner層の初期化
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def set_network(self):
        # 訓練モードに設定
        self.net.train()

        # TransformerBlockモジュールを初期化実行
        self.net.net3_1.apply(self.weights_init)
        self.net.net3_2.apply(self.weights_init)
        print("ネットワーク設定完了")

        # 損失関数の設定
        self.criterion = nn.CrossEntropyLoss()
        # nn.LogSoftmax()を計算してからnn.NLLLoss(negative log likelihood loss)を計算

        # 最適化手法の設定
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.learning_rate)

        return self.criterion, self.optimizer

    def train_model(
        self, dataloaders_dict, num_epochs=3,
    ):

        self.set_network()

        self.max_state = {"epoch_acc_max_net": self.net, "max_acc_epoch": 0}
        epoch_acc_max_net = self.net
        max_acc_epoch = 0
        max_acc = 0
        min_loss = 10 ** 6
        list_acc = []
        list_loss = []
        self.val_state = {}
        sum_time = 0

        # GPUが使えるかを確認
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        print("使用デバイス：", device)
        print("-----start-------")
        # ネットワークをGPUへ
        self.net.to(device)

        # ネットワークがある程度固定であれば、高速化させる
        torch.backends.cudnn.benchmark = True

        batch_size = dataloaders_dict["train"].batch_size

        # epochのループ
        for epoch in range(num_epochs):
            # epochごとの訓練と検証のループ
            for phase in ["train", "val"]:
                if phase == "train":
                    self.net.train()  # モデルを訓練モードに
                else:
                    self.net.eval()  # モデルを検証モードに

                epoch_loss = 0.0  # epochの損失和
                epoch_corrects = 0  # epochの正解数

                iteration = 1

                # 開始時刻を保存
                t_epoch_start = time.time()
                t_iter_start = time.time()

                # データローダーからミニバッチを取り出すループ
                for batch in dataloaders_dict[phase]:
                    # batchはTextとLableの辞書オブジェクト

                    # GPUが使えるならGPUにデータを送る
                    inputs = batch.Text[0].to(device)  # 文章
                    labels = batch.Label.to(device)  # ラベル

                    # optimizerを初期化
                    self.optimizer.zero_grad()

                    # 順伝搬（forward）計算
                    with torch.set_grad_enabled(phase == "train"):

                        # mask作成
                        input_pad = 1  # 単語のIDにおいて、'<pad>': 1 なので
                        input_mask = inputs != input_pad

                        # Transformerに入力
                        outputs, _, _ = self.net(inputs, input_mask)
                        loss = self.criterion(outputs, labels)  # 損失を計算

                        _, preds = torch.max(
                            outputs, 1
                        )  # ラベルを予測（dim=1 列方向のＭａｘを取得、predsは最大のindex）

                        # 訓練時はバックプロパゲーション
                        if phase == "train":
                            loss.backward()  # 損失の計算
                            self.optimizer.step()  # 勾配の更新

                            if iteration % 10 == 0:  # 10iterに1度、lossを表示
                                t_iter_finish = time.time()
                                duration = t_iter_finish - t_iter_start
                                acc = (
                                    torch.sum(preds == labels.data)
                                ).double() / batch_size

                                sum_time += duration
                                print(
                                    "イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec. || 本イテレーションの正解率：{}".format(
                                        iteration, loss.item(), duration, acc
                                    )
                                )
                                t_iter_start = time.time()

                        iteration += 1

                        # 結果の計算
                        # lossの合計を更新
                        epoch_loss += loss.item() * inputs.size(0)
                        # 正解数の合計を更新
                        epoch_corrects += torch.sum(preds == labels.data)

                # epochごとのlossと正解率
                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = epoch_corrects.double() / len(
                    dataloaders_dict[phase].dataset
                )

                if phase == "val":
                    list_acc.append(epoch_acc)
                    list_loss.append(epoch_loss)

                if phase == "val" and min_loss >= epoch_loss:
                    max_acc_epoch = epoch + 1
                    max_acc = epoch_acc
                    min_loss = epoch_loss
                    epoch_acc_max_net = self.net
                    self.max_state["max_acc_epoch"] = epoch + 1
                    self.max_state["epoch_acc_max_net"] = self.net
                    self.max_state["max_acc"] = epoch_acc
                    self.max_state["min_loss"] = epoch_loss

                print(
                    "Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}".format(
                        epoch + 1, num_epochs, phase, epoch_loss, epoch_acc
                    )
                )

        self.val_state["list_acc"] = list_acc
        self.val_state["list_loss"] = list_loss
        self.val_state["epoch"] = num_epochs
        hour, minute, secound = SubFunc().convert_time(sum_time_sec=sum_time)
        print("総学習時間　: {}時間 {}分 {}秒".format(hour, minute, secound))
        print(
            "max_acc_epoch : {} , max_acc : {} , min_loss : {} ".format(
                max_acc_epoch, max_acc, min_loss
            )
        )
        self.net = epoch_acc_max_net

        SubFunc().plot_acc_and_loss(num_epochs, self.val_state)

        return self.net, self.max_state, self.val_state


class SubFunc(object):
    def __init__(self):
        pass

    def convert_time(self, sum_time_sec):
        hour = int(sum_time_sec / 3600)
        minute = int((sum_time_sec - 3600 * hour) / 60)
        secound = int(sum_time_sec - 3600 * hour - 60 * minute)
        return hour, minute, secound

    def plot_acc_and_loss(self, num_epochs, val_state):
        # データ生成
        x = np.linspace(0, num_epochs, num_epochs)

        y_acc = []
        y_loss = []
        for i in range(num_epochs):
            y_acc.append(val_state["list_acc"][i])
            y_loss.append(val_state["list_loss"][i])

        # プロット
        plt.plot(x, y_acc, label="acc")
        plt.plot(x, y_loss, label="loss")

        # 凡例の表示
        plt.legend()

        # プロット表示(設定の反映)
        plt.show()
