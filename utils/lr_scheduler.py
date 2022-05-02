# coding=utf-8
# https://github.com/mzhaoshuai/Divide-and-Co-training/blob/main/utils/lr_scheduler.py
import math


class lr_scheduler(object):
    """learning rate scheduler
    step mode:      ```lr = init_lr * 0.1 ^ {floor(epoch-1 / lr_step)}```
    cosine mode:    ```lr = init_lr * 0.5 * (1 + cos(iter/maxiter))```
    poly mode:      ```lr = init_lr * (1 - iter/maxiter) ^ 0.9```
    HTD mode:       ```lr = init_lr * 0.5 * (1 - tanh(low + (up - low) * iter/maxiter)```
                        https://arxiv.org/pdf/1806.01593.pdf

    Args:
        init_lr:            initial learnig rate.
        mode:               ['cos', 'poly', 'HTD', 'step'].
        num_epochs:         the number of epochs.
        iters_per_epoch:    iterations per epochs.
        lr_milestones:      lr milestones used for 'step' lr mode
        lr_step:            lr step used for 'step' lr mode.
                            It only works when lr_milestones is None.
        lr_step_multiplier: lr multiplier for 'step' lr mode.
        multiplier:         lr multiplier for params group in optimizer.
                            It only works for {3rd, 4th..} groups
        end_lr:             minimal learning rate.
        lower_bound,
        upper_bound:        bound of HTD learning rate strategy.
    """
    def __init__(self, mode='cos',
                        init_lr=0.1,
                        all_iters=300,
                        lr_milestones=None,
                        lr_step=100,
                        lr_step_multiplier=0.1,
                        slow_start_iters=0,
                        slow_start_lr=1e-8,
                        end_lr=1e-8,
                        lower_bound=-6.0,
                        upper_bound=3.0,
                        weight_decay=1e-4):
        assert mode in ['cos', 'poly', 'HTD', 'step']
        self.init_lr = init_lr
        self.now_lr = self.init_lr
        self.end_lr = end_lr
        self.mode = mode
        self.slow_start_iters = slow_start_iters
        self.slow_start_lr = slow_start_lr
        self.total_iters = all_iters - self.slow_start_iters

        # step mode
        self.lr_step = lr_step
        self.lr_milestones = lr_milestones
        self.lr_step_multiplier = lr_step_multiplier
        
        # hyperparameters for HTD
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        # optional for some policies include a weight_decay choice
        self.weight_decay = weight_decay

        # log info
        print("INFO>>> (lr) Using {} learning rate scheduler with"
                " warm-up iterations of {}!".format(self.mode, self.slow_start_iters))

    def __call__(self, optimizer, i=None, epoch=None, global_step=None):
        """call method"""
        T = (epoch * self.iters_per_epoch + i) if global_step is None else global_step

        if self.slow_start_iters > 0 and T <= self.slow_start_iters:
            # slow start strategy -- warm up
            # see   https://arxiv.org/pdf/1812.01187.pdf
            #   Bag of Tricks for Image Classification with Convolutional Neural Networks
            # for details.
            lr = (1.0 * T / self.slow_start_iters) * (self.init_lr - self.slow_start_lr)
            lr = min(lr + self.slow_start_lr, self.init_lr)
        
        elif self.mode == 'cos':
            T = T - self.slow_start_iters
            lr = 0.5 * self.init_lr * (1.0 + math.cos(1.0 * T / self.total_iters * math.pi))
        
        elif self.mode == 'poly':
            T = T - self.slow_start_iters
            lr = self.init_lr * pow(1.0 - 1.0 * T / self.total_iters, 0.9)

        elif self.mode == 'HTD':
            """
            Stochastic Gradient Descent with Hyperbolic-Tangent Decay on Classification.
            https://arxiv.org/pdf/1806.01593.pdf
            """
            T = T - self.slow_start_iters
            ratio = 1.0 * T / self.total_iters
            lr = 0.5 * self.init_lr * (1.0 - math.tanh(
                            self.lower_bound + (self.upper_bound - self.lower_bound) * ratio))

        elif self.mode == 'step':
            if self.lr_milestones is None:
                lr = self.init_lr * (self.lr_step_multiplier ** (epoch // self.lr_step))
            else:
                j = 0
                for mile in self.lr_milestones:
                    if epoch < mile:
                        continue
                    else:
                        j += 1
                lr = self.init_lr * (self.lr_step_multiplier ** j)
        
        else:
            raise NotImplementedError

        lr = max(lr, self.end_lr)
        self.now_lr = lr

        # adjust learning rate
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        """adjust the leaning rate"""
        # these networks uses similar lr policies
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = self.weight_decay * param_group['decay_mult']
