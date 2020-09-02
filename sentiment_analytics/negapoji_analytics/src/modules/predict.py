
import math
import torch

from .make_dataloader import Make_DataLoader


class Predict(object):
    def __init__(self):
        self.np_score = 0

    def soft_max(self, n, p):
        sum_exp_np = math.exp(n)+math.exp(p)
        changed_n = math.exp(n)/sum_exp_np
        changed_p = math.exp(p)/sum_exp_np
        return changed_n, changed_p

    def log_yudo_ratio(self, n, p):
        win_np = max(n, p)
        loose_np = min(n, p)
        changed_n, changed_p = self.soft_max(n, p)
        score = 0

        if win_np == n:
            score = math.log(changed_n/changed_p)
        else:
            score = math.log(changed_p/changed_n)

        return score

    def predict_score(self, net_trained, text_list, TEXT_pkl_path=None):
        pred_str = ""
        np_score_dict = {}
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        net_trained.eval()  # モデルを検証モードに
        net_trained.to(device)  # GPUが使えるならGPUへ送る

        for i in range(len(text_list)):
            try:
                text = text_list[i]
                text = Make_DataLoader().tokenizer_with_preprocessing(text)
                text.insert(0, "<cls>")
                text.append("<eos>")
                text = Make_DataLoader().create_tensor(text_list[i], 256, TEXT_pkl_path=TEXT_pkl_path)
                text = text.unsqueeze_(0)

                # GPUが使えるならGPUにデータを送る
                input = text.to(device)

                # mask作成
                input_pad = 1  # 単語のIDにおいて、'<pad>': 1 なので
                input_mask = input != input_pad

                outputs, _, _ = net_trained(input, input_mask)
                np, pred = torch.max(outputs, 1)  # ラベルを予測
                n_point, p_point = outputs[0]

                np_score = self.log_yudo_ratio(n=n_point, p=p_point)

                print("")
                print(text_list[i])
                # print(n_point, p_point)
                if int(pred) == 0:
                    pred_str = "Negative"
                else:
                    pred_str = "Positive"

                print(pred_str)
                print("{}度 : {}".format(pred_str, np_score))
                np_score_dict[i] = [pred_str, np_score]

            except IndexError:
                print(
                    "IndexError: index 256 is out of bounds for dimension 0 with size 256.  so continued"
                )
                continue

        return np_score_dict
