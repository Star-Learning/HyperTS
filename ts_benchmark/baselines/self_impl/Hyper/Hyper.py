import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
import geoopt

from ts_benchmark.baselines.self_impl.Hyper.Hyper_model import (
    HyperModel,
)
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.baselines.utils import train_val_split


DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "win_size": 100,
    "lr": 0.0001,
    "d_in": 8,
    "d_hidden": 64,
    "d_out": 8,
    "layers": 2,
    "c": 1.0,
    "head_euclidean": True,
    "num_epochs": 10,
    "batch_size": 256,
    "patience": 3,
    "k": 3,
    "lradj": "type1",
    "pct_start": 0.3,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate if epoch < 20 else args.learning_rate * (0.5 ** ((epoch // 20) // 1))}
    elif args.lradj == 'type5':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * (0.5 ** ((epoch // 10) // 1))}
    elif args.lradj == 'type6':
        lr_adjust = {20: args.learning_rate * 0.5 , 40: args.learning_rate * 0.01, 60:args.learning_rate * 0.01,8:args.learning_rate * 0.01,100:args.learning_rate * 0.01 }
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))




class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.check_point = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss



class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.seq_len

    @property
    def learning_rate(self):
        return self.lr
    

class Hyper:
    def __init__(self, **kwargs):
        super(Hyper, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.win_size = self.config.win_size

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {}

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name
    

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception as ignore:
            freq = 'S'
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def detect_validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for input, _ in valid_data_loader:
                input = input.to(device)

                output = self.model(input)

                output = output[:, :, :]

                output = output.detach().cpu()
                true = input.detach().cpu()

                loss = criterion(output, true).detach().cpu().numpy()
                total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Train the model.

        :param train_data: Time series data used for training.
        """

        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.config.c_in = train_data.shape[1]
        self.model = HyperModel(self.config)
        self.model.to(self.device)

        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )

        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)

        train_steps = len(self.train_data_loader)
        main_params = [param for name, param in self.model.named_parameters() if 'mask_generator' not in name]

        # self.optimizer = torch.optim.Adam(main_params,
        #                                   lr=self.config.lr)
        
        self.optimizer = geoopt.optim.RiemannianAdam(main_params, lr=self.config.lr)

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.lr,
        )
 
        time_now = time.time()

        for epoch in range(self.config.num_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            self.model.train()

            step = min(int(len(self.train_data_loader) / 10), 100)
            for i, (input, target) in enumerate(self.train_data_loader):
                iter_count += 1
                self.optimizer.zero_grad()

                input = input.float().to(self.device)
                output = self.model(input)
                output = output[:, :, :]
                loss = self.criterion(output, input)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | training time loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                            (self.config.num_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            valid_loss = self.detect_validate(self.valid_data_loader, self.criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss
                )
            )

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, scheduler, epoch + 1, self.config)

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        self.model.to(self.device)
        self.model.eval()
        self.temp_anomaly_criterion = nn.MSELoss(reduce=False)
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.thre_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

                print(
                    "\t testing time loss: {0} | \n".format(
                        temp_score.detach().cpu().numpy()[0, :5]
                    )
                )

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.test_data_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
        )

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        attens_energy = []

        self.model.to(self.device)
        self.model.eval()
        self.temp_anomaly_criterion = nn.MSELoss(reduce=False)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.train_data_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score).detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.test_data_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.thre_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

                print(
                    "\t testing time loss: {0} | \n".format(
                        temp_score.detach().cpu().numpy()[0, :5]
                    )
                )

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy
