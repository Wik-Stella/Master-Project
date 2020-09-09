import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv = nn.Conv1d(1, 1, 10)
        self.lin0 = nn.Linear(10, 100)
        self.lin0_1 = nn.Linear(100, 100)
        self.lin0_2 = nn.Linear(100, 1)
        self.lin1 = nn.Linear(10, 100)
        self.lin2 = nn.Linear(100, 100)
        self.lin3 = nn.Linear(100, 2)

    def forward(self, input):
        # input = input.view(-1, 1, 10)
        # output = self.conv(input)
        output = self.lin0(input)
        output = F.relu(self.lin0_1(output))
        output = F.relu(self.lin0_2(output))

        output = output.squeeze()
        # output = output.view(-1, 10)
        output = F.relu(self.lin1(output))
        output = F.relu(self.lin2(output))
        output = F.relu(self.lin3(output))
        return output


class Trainer:
    def __init__(self, model):
        self.model = model
        self.init_lr = 0.001
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.init_lr)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=100, eta_min=1e-4)
        self.writer = SummaryWriter(log_dir='runs/adjacency_matrix')

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
            corr_pro = num_correct/len(labels_train)
            print(f'Accuracy of train datasets:  {corr_pro*100}%')

            eval_outputs = self.model(input_eval)
            eval_loss = self.criterion(eval_outputs, labels_eval)
            _, _indices = eval_outputs.max(dim=1)
            _num_correct = labels_eval.eq(_indices).sum().item()
            _corr_pro = _num_correct / len(labels_eval)
            print(f'Accuracy of test datasets:  {_corr_pro*100}%')
            # print(self.scheduler.get_last_lr())
            # self.writer.add_scalar('lr', self.scheduler.get_last_lr()[-1], epoch_i)
            self.writer.add_scalar('Loss/train', loss.item(), epoch_i)
            self.writer.add_scalar('Loss/test', eval_loss, epoch_i)
            self.writer.add_scalar('Accuracy/train', corr_pro, epoch_i)
            self.writer.add_scalar('Accuracy/test', _corr_pro, epoch_i)
        dummy_input = torch.rand(20, 10, 10)
        # with SummaryWriter(comment='LeNet') as w:
        self.writer.add_graph(model, (dummy_input,))
        # self.writer.add_embedding()
        self.writer.close()


if __name__ == '__main__':
    model = Model()
    import numpy as np
    # np.random.seed(1)
    
    # prepare the training datasets
    train_non_euler = np.load('data/train_non_eulerian_adj.npy')
    train_euler = np.load('data/train_eulerian_adj.npy')
    train = np.concatenate((train_euler, train_non_euler))
    # np.random.shuffle(train)
    labels_train = np.concatenate((np.ones(1000), np.zeros(1000)))
    # np.random.shuffle(labels_train)
    labels_train = torch.from_numpy(labels_train).to(torch.int64)
    # print(train.shape)
    input_train = torch.from_numpy(train).to(torch.float32)
    
    # prepare the test datasets
    eval_non_euler = np.load('data/eval_non_eulerian_adj.npy')
    eval_euler = np.load('data/eval_eulerian_adj.npy')
    eval = np.concatenate((eval_euler, eval_non_euler))
    # np.random.shuffle(eval)
    labels_eval = np.concatenate((np.ones(200), np.zeros(200)))
    # np.random.shuffle(labels_eval)
    labels_eval = torch.from_numpy(labels_eval).to(torch.int64)
    # print(eval.shape)
    input_eval = torch.from_numpy(eval).to(torch.float32)

    # Train
    model = Model()
    trainer = Trainer(model)
    trainer.train(input_train, input_eval, labels_train, labels_eval)
