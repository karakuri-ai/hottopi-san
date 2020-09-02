import torch

# from IPython.display import HTML, display

from .make_dataloader import Make_DataLoader
from .predict import Predict


class Look_Attension(object):
    def __init__(self):
        pass

    def highlight(self, word, attn):
        "Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数"

        html_color = "#%02X%02X%02X" % (
            255,
            int(255 * (1 - attn)),
            int(255 * (1 - attn)),
        )
        return '<span style="background-color: {}"> {}</span>'.format(html_color, word)

    def mk_html(self, input, outputs, normlized_weights_1, normlized_weights_2, text_i):
        "HTMLデータを作成する"

        # indexの結果を抽出
        index = 0
        # 文章  #  torch.Size([1, 256])  > torch.Size([256])
        sentence = input.squeeze_(0)
        TEXT = Make_DataLoader().pickle_load_TEXT()

        # indexのAttentionを抽出と規格化
        attens1 = normlized_weights_1[index, 0, :]  # 0番目の<cls>のAttention
        attens1 /= attens1.max()

        attens2 = normlized_weights_2[index, 0, :]  # 0番目の<cls>のAttention
        attens2 /= attens2.max()

        np, pred = torch.max(outputs, 1)  # ラベルを予測
        n_point, p_point = outputs[0]

        np_score = Predict().log_yudo_ratio(n=n_point, p=p_point)

        if pred == 0:
            pred_str = "Negative"
        else:
            pred_str = "Positive"

        # 表示用のHTMLを作成する
        html = "{}<br>".format(text_i)
        html += "推論ラベル：{}<br>".format(pred_str)
        html += "{}度：{}<br>".format(pred_str, np_score)

        # 1段目のAttention
        html += "[TransformerBlockの1段目のAttentionを可視化]<br>"
        for word, attn in zip(sentence, attens1):
            html += self.highlight(TEXT.vocab.itos[word], attn)
        html += "<br><br>"

        # 2段目のAttention
        html += "[TransformerBlockの2段目のAttentionを可視化]<br>"
        for word, attn in zip(sentence, attens2):
            html += self.highlight(TEXT.vocab.itos[word], attn)

        html += "<br><br><br>"

        return html

    def predict_with_attension(self, net_trained, text_list):

        device = torch.device("cpu")
        net_trained.eval()  # モデルを検証モードに
        net_trained.to(device)

        html_output = ""

        # インプットデータ
        for i in range(len(text_list)):
            text = text_list[i]
            text = Make_DataLoader().tokenizer_with_preprocessing(text)
            text.insert(0, "<cls>")
            text.append("<eos>")
            text = Make_DataLoader().create_tensor(text_list[i], 256)
            text = text.unsqueeze_(0)

            # GPUが使えるならGPUにデータを送る
            input = text.to(device)
            # mask作成
            input_pad = 1  # 単語のIDにおいて、'<pad>': 1 なので
            input_mask = input != input_pad
            # print(input)
            # print(input_mask)

            outputs, normlized_weights_1, normlized_weights_2 = net_trained(
                input, input_mask
            )
            _, preds = torch.max(outputs, 1)  # ラベルを予測

            html_output += self.mk_html(
                input,
                outputs,
                normlized_weights_1,
                normlized_weights_2,
                text_i=text_list[i],
            )  # HTML作成

        return html_output
