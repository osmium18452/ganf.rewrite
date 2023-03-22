import os

import numpy
import numpy as np
import torch
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm

from models.GANF import GANF


class Ganf:
    def __init__(self, n_sensor, cuda, n_blocks, hidden_size, hidden_layers, batch_norm):
        init = torch.zeros([n_sensor, n_sensor])
        init = xavier_uniform_(init).abs()
        init = init.fill_diagonal_(0.0)
        if cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.A = torch.tensor(init, requires_grad=True, device=device)
        # self.A=init.clone().requires_grad_(True).to(device)
        self.n_sensor = n_sensor
        self.model = GANF(n_blocks, 1, hidden_size, hidden_layers, dropout=0.0, batch_norm=batch_norm)
        self.model.to(torch.device('cuda'))
        self.c = 1.0
        self.lamda = 0.

    def tune_matrix_A(self, train_set, learning_rate, batch_size, weight_decay, cuda):
        # tune matrix A
        h_A_old = np.inf
        max_iter = 20
        rho_max = 1e16
        h_tol = 1e-4
        main_pbar = tqdm(total=max_iter, ascii=True)
        main_pbar.set_description('tuning matrix A')
        for _ in range(max_iter):
            while self.c < rho_max:
                lr = learning_rate
                optimizer = torch.optim.Adam([
                    {'params': self.model.parameters(), 'weight_decay': weight_decay},
                    {'params': [self.A]},
                ], lr=lr, weight_decay=0.0)

                self.model.train()
                iters = train_set.shape[0] // batch_size
                pbar = tqdm(total=iters, ascii=True, leave=False)
                pbar.set_description('training...')
                for i in range(iters):
                    batch_x = train_set[i * batch_size:(i + 1) * batch_size]
                    if cuda:
                        batch_x = batch_x.cuda()
                    optimizer.zero_grad()
                    loss = -self.model(batch_x, self.A)
                    h = torch.trace(torch.matrix_exp(self.A * self.A)) - self.n_sensor
                    total_loss = loss + 0.5 * self.c * h * h + self.lamda * h
                    total_loss.backward()
                    clip_grad_value_(self.model.parameters(), 1)
                    optimizer.step()
                    self.A.data.copy_(torch.clamp(self.A.data, min=0, max=1))
                    pbar.update(1)
                    pbar.set_postfix_str('h:%.5e, loss:%.5e, c:%5e' % (h, total_loss,self.c))
                pbar.close()
                if iters * batch_size != train_set.shape[0]:
                    batch_x = train_set[iters * batch_size:]
                    if cuda:
                        batch_x = batch_x.cuda()
                    optimizer.zero_grad()
                    loss = -self.model(batch_x, self.A)
                    h = torch.trace(torch.matrix_exp(self.A * self.A)) - self.n_sensor
                    total_loss = loss + 0.5 * self.c * h * h + self.lamda * h
                    total_loss.backward()
                    clip_grad_value_(self.model.parameters(), 1)
                    optimizer.step()
                    self.A.data.copy_(torch.clamp(self.A.data, min=0, max=1))

                del optimizer
                torch.cuda.empty_cache()
                h = torch.trace(torch.matrix_exp(self.A * self.A)) - self.n_sensor
                if h.item() > 0.5 * h_A_old:
                    self.c *= 10
                else:
                    break
            h = torch.trace(torch.matrix_exp(self.A * self.A)) - self.n_sensor
            h_A_old = h.item()
            self.lamda += self.c * h.item()
            main_pbar.update(1)
            main_pbar.set_postfix_str('h:%.5e' % h.item())
            if h_A_old <= h_tol or self.c >= rho_max:
                break

    def train_model(self, train_set, learning_rate, batch_size, weight_decay, epoch, cuda):
        # train model
        lr = learning_rate
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'weight_decay': weight_decay},
            {'params': [self.A]},
        ], lr=lr, weight_decay=0.0)
        self.model.train()
        main_pbar = tqdm(total=epoch, ascii=True)
        main_pbar.set_description('training GANF')
        for e in range(epoch):
            iters = train_set.shape[0] // batch_size
            pbar = tqdm(total=iters, ascii=True, leave=False)
            pbar.set_description('training...')
            for i in range(iters):
                batch_x = train_set[i * batch_size:(i + 1) * batch_size]
                if cuda:
                    batch_x = batch_x.cuda()
                optimizer.zero_grad()
                loss = -self.model(batch_x, self.A)
                h = torch.trace(torch.matrix_exp(self.A * self.A)) - self.n_sensor
                total_loss = loss + 0.5 * self.c * h * h + self.lamda * h
                total_loss.backward()
                clip_grad_value_(self.model.parameters(), 1)
                optimizer.step()
                self.A.data.copy_(torch.clamp(self.A.data, min=0, max=1))
                pbar.update(1)
                pbar.set_postfix_str('loss:%.5e, h:%.5e' % (total_loss, h.item()))
            pbar.close()
            if iters * batch_size != train_set.shape[0]:
                batch_x = train_set[iters * batch_size:]
                if cuda:
                    batch_x = batch_x.cuda()
                optimizer.zero_grad()
                loss = -self.model(batch_x, self.A)
                h = torch.trace(torch.matrix_exp(self.A * self.A)) - self.n_sensor
                total_loss = loss + 0.5 * self.c * h * h + self.lamda * h
                total_loss.backward()
                clip_grad_value_(self.model.parameters(), 1)
                optimizer.step()
                self.A.data.copy_(torch.clamp(self.A.data, min=0, max=1))
            main_pbar.update()
            main_pbar.set_postfix_str('loss:%.5e, h:%.5e' % (loss.item(), h.item()))

    def evaluate(self, test_set, batch_size, cuda):
        # evaluate model
        loss_list=[]
        self.model.eval()
        iters = test_set.shape[0] // batch_size
        pbar = tqdm(total=iters, ascii=True)
        pbar.set_description('evaluating...')
        for i in range(iters):
            batch_x = test_set[i * batch_size:(i + 1) * batch_size]
            if cuda:
                batch_x = batch_x.cuda()
            # print(self.model.test(batch_x,self.A.data))
            # exit()
            loss = -self.model.test(batch_x, self.A.data).cpu().detach().numpy()
            loss_list.append(loss)
            pbar.update(1)
            pbar.set_postfix_str('loss:%.5e' % loss.mean().item())
        pbar.close()
        if iters * batch_size != test_set.shape[0]:
            batch_x = test_set[iters * batch_size:]
            if cuda:
                batch_x = batch_x.cuda()
            loss = -self.model.test(batch_x, self.A.data).cpu().detach().numpy()
            loss_list.append(loss)
        return np.concatenate(loss_list)
