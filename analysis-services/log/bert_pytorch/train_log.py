from torch.utils.data import DataLoader
from bert_pytorch.model import BERT
from bert_pytorch.trainer import BERTTrainer
from bert_pytorch.dataset import LogDataset, WordVocab
from bert_pytorch.dataset.sample import generate_train_valid
from bert_pytorch.dataset.utils import save_parameters

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import tqdm
import gc
import os

class Trainer():
    def __init__(self, options):
        defaults = {
            "device": 'cuda' if torch.cuda.is_available() else 'cpu',
            "output_dir": "./output/",
            "model_dir": "./output/bert/",
            "model_path": "./output/bert/best_bert.pth",
            "train_vocab": "./output/train",
            "vocab_path": "./output/vocab.pkl",

            "window_size": 128,
            "adaptive_window": True,
            "seq_len": 100,
            "max_len": 512,
            "min_len": 10,
            "mask_ratio": 0.35,
            "train_ratio": 0.6,
            "valid_ratio": 0.15,
            "test_ratio": 0.25,

            "is_logkey": True,
            "is_time": False,

            "hypersphere_loss": True,
            "hypersphere_loss_test": True,

            "scale": None,
            "scale_path": None,

            "hidden": 64,
            "layers": 4,
            "attn_heads": 4,

            "epochs": 50,
            "n_epochs_stop": 10,
            "batch_size": 32,

            "corpus_lines": None,
            "on_memory": True,
            "num_workers": 5,

            "lr": 1e-3,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_weight_decay": 0.00,
            "with_cuda": True,
            "cuda_devices": None,
            "log_freq": None
        }

        # Merge defaults with incoming options
        config = {**defaults, **options}

        # Assign to self
        for key, value in config.items():
            setattr(self, key, value)

        print("Save options parameters")
        save_parameters(options, self.model_dir + "parameters.txt")

    def train(self):

        print("Loading vocab", self.vocab_path)
        vocab = WordVocab.load_vocab(self.vocab_path)
        print("vocab Size: ", len(vocab))

        print("\nLoading Train Dataset")
        logkey_train, logkey_valid, time_train, time_valid = generate_train_valid(self.output_path + "train", window_size=self.window_size,
                                     adaptive_window=self.adaptive_window,
                                     valid_size=self.valid_ratio,
                                     sample_ratio=self.sample_ratio,
                                     scale=self.scale,
                                     scale_path=self.scale_path,
                                     seq_len=self.seq_len,
                                     min_len=self.min_len
                                    )

        train_dataset = LogDataset(logkey_train,time_train, vocab, seq_len=self.seq_len,
                                    corpus_lines=self.corpus_lines, on_memory=self.on_memory, mask_ratio=self.mask_ratio)

        print("\nLoading valid Dataset")

        valid_dataset = LogDataset(logkey_valid, time_valid, vocab, seq_len=self.seq_len, on_memory=self.on_memory, mask_ratio=self.mask_ratio)

        print("Creating Dataloader")
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                      collate_fn=train_dataset.collate_fn, drop_last=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                       collate_fn=train_dataset.collate_fn, drop_last=True)
        del train_dataset
        del valid_dataset
        del logkey_train
        del logkey_valid
        del time_train
        del time_valid
        gc.collect()

        if os.path.isfile(self.model_path):
            print("Loading Pre-Trained BERT model")
            model = torch.load(self.model_path)
            bert = model.bert
        else:
            print("Building BERT model")
            bert = BERT(len(vocab), max_len=self.max_len, hidden=self.hidden, n_layers=self.layers, attn_heads=self.attn_heads,
                        is_logkey=self.is_logkey, is_time=self.is_time)

        print("Creating BERT Trainer")
        self.trainer = BERTTrainer(bert, len(vocab), train_dataloader=self.train_data_loader, valid_dataloader=self.valid_data_loader,
                              lr=self.lr, betas=(self.adam_beta1, self.adam_beta2), weight_decay=self.adam_weight_decay,
                              with_cuda=self.with_cuda, cuda_devices=self.cuda_devices, log_freq=self.log_freq,
                              is_logkey=self.is_logkey, is_time=self.is_time,
                              hypersphere_loss=self.hypersphere_loss)

        self.start_iteration(surfix_log="log2")

        self.plot_train_valid_loss("_log2")

    def start_iteration(self, surfix_log):
        print("Training Start")
        best_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(self.epochs):
            print("\n")
            if self.hypersphere_loss:
                center = self.calculate_center([self.train_data_loader, self.valid_data_loader])
                # center = self.calculate_center([self.train_data_loader])
                self.trainer.hyper_center = center

            _, train_dist = self.trainer.train(epoch)
            avg_loss, valid_dist = self.trainer.valid(epoch)
            self.trainer.save_log(self.model_dir, surfix_log)

            if self.hypersphere_loss:
                self.trainer.radius = self.trainer.get_radius(train_dist + valid_dist, self.trainer.nu)

            # save model after 10 warm up epochs
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.trainer.save(self.model_path)
                epochs_no_improve = 0

                if epoch > 10 and self.hypersphere_loss:
                    best_center = self.trainer.hyper_center
                    best_radius = self.trainer.radius
                    total_dist = train_dist + valid_dist

                    if best_center is None:
                        raise TypeError("center is None")

                    print("best radius", best_radius)
                    best_center_path = self.model_dir + "best_center.pt"
                    print("Save best center", best_center_path)
                    torch.save({"center": best_center, "radius": best_radius}, best_center_path)

                    total_dist_path = self.model_dir + "best_total_dist.pt"
                    print("save total dist: ", total_dist_path)
                    torch.save(total_dist, total_dist_path)
            else:
                epochs_no_improve += 1

            if epochs_no_improve == self.n_epochs_stop:
                print("Early stopping")
                break

    def calculate_center(self, data_loader_list):
        print("start calculate center")
        with torch.no_grad():
            outputs = 0
            total_samples = 0
            for data_loader in data_loader_list:
                totol_length = len(data_loader)
                data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
                for i, data in data_iter:
                    data = {key: value.to(self.device) for key, value in data.items()}

                    result = self.trainer.model.forward(data["bert_input"], data["time_input"])
                    cls_output = result["cls_output"]

                    outputs += torch.sum(cls_output.detach().clone(), dim=0)
                    total_samples += cls_output.size(0)

        center = outputs / total_samples

        return center

    def plot_train_valid_loss(self, surfix_log):
        train_loss = pd.read_csv(self.model_dir + f"train{surfix_log}.csv")
        valid_loss = pd.read_csv(self.model_dir + f"valid{surfix_log}.csv")
        sns.lineplot(x="epoch", y="loss", data=train_loss, label="train loss")
        sns.lineplot(x="epoch", y="loss", data=valid_loss, label="valid loss")
        plt.title("epoch vs train loss vs valid loss")
        plt.legend()
        plt.savefig(self.model_dir + "train_valid_loss.png")
        plt.show()
        print("plot done")