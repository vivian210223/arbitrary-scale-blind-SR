import torch
import torch.nn as nn

@torch.no_grad()
class dequeue_and_enqueue(nn.Module):
    """It is the training pair pool for increasing the diversity in a batch.
    Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
    batch could not have different resize scaling factors. Therefore, we employ this training pair pool
    to increase the degradation diversity in a batch.
    """
    def __init__(self, config, state):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__()
        self.b, c, h, w = config['total_batch_size'], 3 , config['inp_size'], config['inp_size']
        self.queue_size = config['queue_size']
        self.state = state
        k = config.get('blur').get('kernel_size')
        if k is None:
            k = 21
        # initialize
        if state == 'degrade':
            assert self.queue_size % self.b == 0, f'queue size {self.queue_size} should be divisible by batch size {self.b}'
            self.queue_q = torch.zeros(self.queue_size, c, h, w).to(self.device)
            self.queue_k = torch.zeros(self.queue_size, c, h, w).to(self.device)
            self.queue_ker = torch.zeros(self.queue_size, 1, k, k).to(self.device)
            self.queue_ptr = 0
        elif state =='SR':
            assert self.queue_size % self.b == 0, f'queue size {self.queue_size} should be divisible by batch size {self.b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).to(self.device)
            self.queue_gt = torch.zeros(self.queue_size, config['sample_q'], c).to(self.device)
            self.queue_cell = torch.zeros(self.queue_size, config['sample_q'], 2).to(self.device)
            self.queue_coord = torch.zeros(self.queue_size, config['sample_q'], 2).to(self.device)
            self.queue_ker = torch.zeros(self.queue_size, 1, k, k).to(self.device)
            self.queue_scale = torch.zeros(self.queue_size).to(self.device)
            self.queue_ptr = 0
        
    def forward(self, inp):
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            if self.state == 'degrade':
                self.queue_q = self.queue_q[idx]
                self.queue_k = self.queue_k[idx]
                self.queue_ker = self.queue_ker[idx]
                # get first b samples
                q_dequeue = self.queue_q[0:self.b, :, :, :].clone()
                k_dequeue = self.queue_k[0:self.b, :, :, :].clone()
                ker_dequeue = self.queue_ker[0:self.b, :, :, :].clone()
                # update the queue
                self.queue_q[0:self.b, :, :, :] = inp['query'].clone()
                self.queue_k[0:self.b, :, :, :] = inp['key'].clone()
                self.queue_ker[0:self.b, :, :, :] = inp['lr_gt_kernel'].clone()
    
                return  q_dequeue, k_dequeue, ker_dequeue

            
            elif self.state == 'SR':
                self.queue_lr = self.queue_lr[idx]
                self.queue_gt = self.queue_gt[idx]
                self.queue_cell = self.queue_cell[idx]
                self.queue_coord = self.queue_coord[idx]
                self.queue_ker = self.queue_ker[idx]
                self.queue_scale = self.queue_scale[idx]
                # get first b samples
                lr_dequeue = self.queue_lr[0:self.b, :, :, :].clone()
                gt_dequeue = self.queue_gt[0:self.b, :, :].clone()
                cell_dequeue = self.queue_cell[0:self.b, :, :].clone()
                coord_dequeue = self.queue_coord[0:self.b, :, :].clone()
                ker_dequeue = self.queue_ker[0:self.b, :, :, :].clone()
                scale_dequeue = self.queue_scale[0:self.b].clone()
                
                # update the queue
                self.queue_lr[0:self.b, :, :, :] = inp['lr'].clone()
                self.queue_gt[0:self.b, :, :] = inp['gt'].clone()
                self.queue_cell[0:self.b, :, :] = inp['cell'].clone()
                self.queue_coord[0:self.b, :, :] = inp['coord'].clone()
                self.queue_ker[0:self.b, :, :, :] = inp['lr_gt_kernel'].clone()
                self.queue_scale[0:self.b] = inp['scale'].clone()
                
                return  lr_dequeue, gt_dequeue, cell_dequeue, coord_dequeue, scale_dequeue.unsqueeze(-1), ker_dequeue
            
            
        else:
            # pool isn't full
            if self.state == 'degrade':
                self.queue_q[self.queue_ptr:self.queue_ptr + self.b, :, :, :] = inp['query'].clone()
                self.queue_k[self.queue_ptr:self.queue_ptr + self.b, :, :, :] = inp['key'].clone()
                self.queue_ker[self.queue_ptr:self.queue_ptr + self.b, :, :, :] = inp['lr_gt_kernel'].clone()
                self.queue_ptr = self.queue_ptr + self.b
                return inp['query'], inp['key'], inp['lr_gt_kernel']

            
            elif self.state == 'SR':
                self.queue_lr[self.queue_ptr:self.queue_ptr + self.b, :, :, :] = inp['lr'].clone()
                self.queue_gt[self.queue_ptr:self.queue_ptr + self.b, :, :] = inp['gt'].clone()
                self.queue_cell[self.queue_ptr:self.queue_ptr + self.b, :, :] = inp['cell'].clone()
                self.queue_coord[self.queue_ptr:self.queue_ptr + self.b, :, :] = inp['coord'].clone()
                self.queue_ker[self.queue_ptr:self.queue_ptr + self.b, :, :, :] = inp['lr_gt_kernel'].clone()
                self.queue_scale[self.queue_ptr:self.queue_ptr + self.b] = inp['scale'].clone()
                self.queue_ptr = self.queue_ptr + self.b
                return inp['lr'], inp['gt'], inp['cell'], inp['coord'], inp['scale'].unsqueeze(-1), inp['lr_gt_kernel']
            
            