"""
This module implements the simplest form of rehearsal training: Experience Replay. It maintains a buffer
of previously seen examples and uses them to augment the current batch during training.

Example usage:
    model = Er(backbone, loss, args, transform, dataset)
    loss = model.observe(inputs, labels, not_aug_inputs, epoch)

"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F
from torch.nn import KLDivLoss
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer

from copy import deepcopy


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))

def kl(old, new):
    log_target = F.log_softmax(old, dim=-1)
    input = F.softmax(new, dim=-1)

    return torch.mean(F.kl_div(input, log_target, log_target=True, reduction="batchmean"))


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def kl_divergence_stable(logits_student, logits_teacher, temperature=1.0):
    """
    Numerically stable KL divergence for knowledge distillation.
    
    Args:
        logits_student (Tensor): Logits from the student network (B x C).
        logits_teacher (Tensor): Logits from the teacher network (B x C).
        temperature (float): Temperature scaling factor.
        
    Returns:
        Tensor: The KL divergence loss.
    """
    # Apply temperature scaling
    scaled_logits_student = logits_student / temperature
    scaled_logits_teacher = logits_teacher / temperature

    # Compute log probabilities in a numerically stable way
    log_p_student = F.log_softmax(scaled_logits_student, dim=-1)
    log_p_teacher = F.log_softmax(scaled_logits_teacher, dim=-1)
    
    # Compute KL divergence using teacher's soft probabilities
    kl_loss = F.kl_div(log_p_student, log_p_teacher.exp(), reduction='batchmean')

    return kl_loss




class ERLwF(ContinualModel):
    """Continual learning via Experience Replay."""
    NAME = 'er_lwf'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the Er model.

        This model requires the `add_rehearsal_args` to include the buffer-related arguments.
        """
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, default=0.5,
                            help='Penalty weight.')
        parser.add_argument('--softmax_temp', type=float, default=2,
                            help='Temperature of the softmax function.')
        parser.add_argument("--ema", type=float, default=0.0, help='Momentum teacher')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        The ER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(ERLwF, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)

        self.gamma = self.args.ema

    def begin_task(self, dataset):

        print(self.n_seen_classes)

        self.old_net = deepcopy(self.net).to(self.device)

        for param in self.old_net.parameters():
            param.requires_grad = False

        self.old_net.train()


    def _update_teacher(self):

        for p_s, p_t in zip(self.net.parameters(), self.old_net.parameters()):

            p_t.data.mul_(self.gamma).add_((1-self.gamma) * p_s.detach().data)

    def observe2(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """

        real_batch_size = inputs.shape[0]
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        self.opt.zero_grad()

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            #inputs = torch.cat((inputs, buf_inputs))
            #labels = torch.cat((labels, buf_labels))
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha* self.loss(buf_outputs, buf_labels)

        
    
        
        logits = self.old_net(inputs)

        
        #print(self.n_seen_classes)
        # distillation
        #and self.current_task>1
        if not self.buffer.is_empty() :
            
            #loss += self.args.alpha * modified_kl_div(smooth(self.soft(logits[real_batch_size:, :self.n_past_classes]).to(self.device), self.args.softmax_temp, 1),
            #                                          smooth(self.soft(outputs[real_batch_size:, :self.n_past_classes]), self.args.softmax_temp, 1))
            buf_inputs_1, buf_inputs, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device, return_not_aug=True, not_aug_transform=self.transform)
            
            logits = self.old_net(buf_inputs_1)
            outputs = self.net(buf_inputs)

            #loss += self.args.alpha/2 * F.mse_loss(outputs, logits)
            #loss += self.args.alpha/3 * modified_kl_div(smooth(self.soft(logits[:,: self.n_seen_classes]), self.args.softmax_temp, 1),
            #                                        smooth(self.soft(outputs[:, :self.n_seen_classes]), self.args.softmax_temp, 1))
            #loss += self.args.alpha/3 * kl(logits[:,: self.n_seen_classes], outputs[:, :self.n_seen_classes])

            loss += self.args.alpha/3 * F.mse_loss(logits, outputs)


        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])
        
        if self.gamma > 0.0:
            self._update_teacher()

        return loss.item()


    def observe2(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """

        real_batch_size = inputs.shape[0]
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        self.opt.zero_grad()

        logits = self.old_net(not_aug_inputs)



        #loss += self.args.alpha * F.mse_loss(outputs, logits)
        #loss += self.args.alpha/4 * modified_kl_div(smooth(self.soft(logits[:,: self.n_seen_classes]), self.args.softmax_temp, 1),
        #                                        smooth(self.soft(outputs[:, :self.n_seen_classes]), self.args.softmax_temp, 1))


        if not self.buffer.is_empty():
            buf_inputs, buf_inputs_2, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device,
                return_not_aug=True)
            
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha* self.loss(buf_outputs, buf_labels)

            logits = self.old_net(buf_inputs_2)
            #loss += self.args.alpha * F.mse_loss(logits, buf_outputs)
            #loss += self.args.alpha/4 * modified_kl_div(smooth(self.soft(logits[:,: self.n_seen_classes]), self.args.softmax_temp, 1),
            #                                        smooth(self.soft(buf_outputs[:, :self.n_seen_classes]), self.args.softmax_temp, 1))




        
    
        

        """
        #print(self.n_seen_classes)
        # distillation
        #and self.current_task>1
        if not self.buffer.is_empty() :
            
            #loss += self.args.alpha * modified_kl_div(smooth(self.soft(logits[real_batch_size:, :self.n_past_classes]).to(self.device), self.args.softmax_temp, 1),
            #                                          smooth(self.soft(outputs[real_batch_size:, :self.n_past_classes]), self.args.softmax_temp, 1))
            buf_inputs_1, buf_inputs, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device, return_not_aug=True, not_aug_transform=self.transform)
            
            logits = self.old_net(buf_inputs_1)
            outputs = self.net(buf_inputs)

            #loss += self.args.alpha/2 * F.mse_loss(outputs, logits)
            #loss += self.args.alpha/3 * modified_kl_div(smooth(self.soft(logits[:,: self.n_seen_classes]), self.args.softmax_temp, 1),
            #                                        smooth(self.soft(outputs[:, :self.n_seen_classes]), self.args.softmax_temp, 1))
            #loss += self.args.alpha/3 * kl(logits[:,: self.n_seen_classes], outputs[:, :self.n_seen_classes])

            loss += self.args.alpha/3 * F.mse_loss(logits, outputs)

        """
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])
        
        if self.gamma > 0.0:
            self._update_teacher()

        return loss.item()
    

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """

        real_batch_size = inputs.shape[0]
        temp = not_aug_inputs

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_inputs_, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device,
                return_not_aug=True)
            inputs = torch.cat((inputs, buf_inputs_))
            temp = torch.cat((not_aug_inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        logits = self.old_net(temp)
        #loss += F.mse_loss(logits, outputs)/4
        #loss += self.args.alpha/4 * modified_kl_div(smooth(self.soft(logits[:,: self.n_seen_classes]), self.args.softmax_temp, 1),
        #                                       smooth(self.soft(outputs[:, :self.n_seen_classes]), self.args.softmax_temp, 1))

        loss += self.args.alpha / 4 * kl_divergence_stable(outputs, logits, self.args.softmax_temp)


        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])
        if self.gamma > 0.0:
            self._update_teacher()

        return loss.item()