import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv = nn.Conv1d(1, 1, 10)
        self.rnn = nn.LSTM(9, 10, bidirectional=True)
        self.lin = nn.Linear(20, 2)

    def forward(self, input):
        lens = [len(i) for i in input]
        input = pad_sequence(input).permute(1, 0, 2)
        output, *_ = self.rnn(input)
        output = [j[lens[num]-1]for num, j in enumerate(output)]
        output = torch.stack(output)
        output = self.lin(output)
        return output


class Trainer:
    def __init__(self, model):
        self.model = model
        self.init_lr = 0.001
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.init_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=100, eta_min=1e-4)
        self.writer = SummaryWriter(log_dir='runs/adjacency_list')

    def train(self, input_train, input_eval, labels_train, labels_eval):
        for epoch_i in range(2000):
            print(f'epoch {epoch_i}')
            outputs = self.model(input_train)
            loss = self.criterion(outputs, labels_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step(epoch_i)

            _, indices = outputs.max(dim=1)
            num_correct = labels_train.eq(indices).sum().item()
            corr_pro = num_correct / len(labels_train)
            print(f'Accuracy of train datasets:  {corr_pro * 100}%')

            eval_outputs = self.model(input_eval)
            eval_loss = self.criterion(eval_outputs, labels_eval)
            _, _indices = eval_outputs.max(dim=1)
            _num_correct = labels_eval.eq(_indices).sum().item()
            _corr_pro = _num_correct / len(labels_eval)
            print(f'Accuracy of test datasets:  {_corr_pro * 100}%')
            # print(self.scheduler.get_last_lr())
            # self.writer.add_scalar('lr', self.scheduler.get_last_lr()[-1], epoch_i)
            self.writer.add_scalar('Loss/train', loss.item(), epoch_i)
            self.writer.add_scalar('Loss/test', eval_loss, epoch_i)
            self.writer.add_scalar('Accuracy/train', corr_pro, epoch_i)
            self.writer.add_scalar('Accuracy/test', _corr_pro, epoch_i)
        dummy_input = torch.rand(20, 10, 2)

        self.writer.add_graph(model, (dummy_input,))

        self.writer.close()


if __name__ == '__main__':
    model = Model()
    import numpy as np

    # np.random.seed(1)

    # prepare the training datasets
    train_non_euler = np.load('data/train_non_eulerian_adj_list.npy', allow_pickle=True)
    train_euler = np.load('data/train_eulerian_adj_list.npy', allow_pickle=True)
    train = np.concatenate((train_euler, train_non_euler))
    # np.random.shuffle(train)
    labels_train = np.concatenate((np.ones(1000), np.zeros(1000)))
    # np.random.shuffle(labels_train)
    labels_train = torch.from_numpy(labels_train).to(torch.int64)
    # print(train.shape)
    input_train = [torch.from_numpy(i).to(torch.float32) for i in train]

    # prepare the test datasets
    eval_non_euler = np.load('data/eval_non_eulerian_adj_list.npy', allow_pickle=True)
    eval_euler = np.load('data/eval_eulerian_adj_list.npy', allow_pickle=True)
    eval = np.concatenate((eval_euler, eval_non_euler))
    # np.random.shuffle(eval)
    labels_eval = np.concatenate((np.ones(200), np.zeros(200)))
    # np.random.shuffle(labels_eval)
    labels_eval = torch.from_numpy(labels_eval).to(torch.int64)
    # print(eval.shape)
    input_eval = [torch.from_numpy(j).to(torch.float32) for j in eval]

    # Train
    model = Model()
    trainer = Trainer(model)
    trainer.train(input_train, input_eval, labels_train, labels_eval)
