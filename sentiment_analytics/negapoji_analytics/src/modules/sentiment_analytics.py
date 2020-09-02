
import torch


from IPython.display import HTML, display

from .classifier import Classifier as BaseClassifier
from .make_dataloader import Make_DataLoader
from .transformer_network import TransformerClassification, Construct_Network
from .predict import Predict
from .look_attension import Look_Attension


class Analytics(BaseClassifier):
    def __init__(self, gpu_id: int = -1, TEXT_pkl_path=None):
        """初期化
        Args:
            TEXT_pkl_path (str or None):
                デフォルトは　None.
                self.TEXT を TEXT.pkl から直接読み込みたい場合は， TEXT.pkl の場所を文字列で指定してください
        """ 
        self.TEXT_pkl_path = TEXT_pkl_path
        if self.TEXT_pkl_path is None:
            datasets_dict, self.TEXT = Make_DataLoader().make_ds_and_TEXT()
            self.dataloaders_dict = Make_DataLoader().make_dl(datasets_dict=datasets_dict)
        else:
            self.TEXT = Make_DataLoader().pickle_load_TEXT(path=self.TEXT_pkl_path)

        self.np = 0

    def fit(
        self, dataloaders_dict, TEXT, num_epochs=1 ,**kwargs
    ):
        self.dataloaders_dict = dataloaders_dict
        self.TEXT = TEXT
        # モデル構築
        self.num_epochs = num_epochs

        # batch = next(iter(self.dataloaders_dict["train"]))

        self.net_trained, max_state, val_state = Construct_Network(
            self.TEXT
        ).train_model(dataloaders_dict=self.dataloaders_dict, num_epochs=self.num_epochs)

        return self.net_trained, max_state, val_state

    def predict_np(self, text_list, save_path="./weights/20202.pth"):

        net_trained = self.load(save_path)
        np_score_dict = Predict().predict_score(
            net_trained=net_trained, text_list=text_list, TEXT_pkl_path=self.TEXT_pkl_path
        )

        return np_score_dict

    def predict_with_attension(self, text_list, save_path="./weights/20202.pth"):

        net_trained = self.load(save_path)
        html_output = Look_Attension().predict_with_attension(net_trained, text_list)
        display(HTML(html_output))

    def save(self, save_path="./weights/20202.pth"):
        torch.save(self.net_trained.state_dict(), save_path)

    def load(self, save_path="./weights/20202.pth"):
        net_trained = TransformerClassification(
            text_embedding_vectors=self.TEXT.vocab.vectors
        )
        net_trained.load_state_dict(torch.load(save_path))
        return net_trained
