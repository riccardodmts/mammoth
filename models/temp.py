    

# ema 0.996 temp 1.0 -> 51.89 overall. plasticity: 89.9, 71.2, 73.05, 76.5, 85.3, 
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



# seeds 1, 10, 100, 1000, 10000
# no different data augment -> 48, 45.78, 50.14, 43.86, 48.93
# with -> 49.2, 48.62, 49.31, 45.83, 49.57