import torch
import torch.nn.functional as F
from torch_geometric.nn import DataParallel
from torch.optim.lr_scheduler import LambdaLR, CyclicLR
from torch_geometric.data import DataLoader
import tqdm
import time
import math
import torch.nn as nn

class CompositeDRNTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        lr_scheduler,
        loss_func=F.binary_cross_entropy,
        #loss_func=nn.BCEWithLogitsLoss(),
        lr_sched_type="Const",
        acc_rate=1,
        category_weights=torch.tensor([1.0, 1.0]),
        parallel=True,
        logger=None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        lr_scheduler=LambdaLR(self.optimizer, lambda epoch:1),
        self.device = device
        self.loss_func = loss_func
        self.lr_scheduler = lr_scheduler
        self.lr_sched = lr_sched_type
        self.acc_rate = acc_rate
        self._category_weights = category_weights.to(device)
        self.parallel = parallel
        self.logger = logger if logger is not None else print
        if self.parallel:
            self.model = DataParallel(self.model,device_ids=list(range(torch.cuda.device_count()))).to(self.device)
            print("NUMBER OF CUDA CORES:",torch.cuda.device_count())

        model_to_count = self.model.module if hasattr(self.model, "module") else self.model
        total_params = sum(p.numel() for p in model_to_count.parameters() if p.requires_grad)
        self.logger(f"Total trainable parameters: {total_params:,}")

    def train_epoch(self, data_loader):
        if '{}'.format(self.device) != 'cpu':
            torch.cuda.reset_max_memory_allocated(self.device)
        self.model.train()

        summary = dict(acc_loss=[], acc_lr=[])
        sum_loss = 0.0
        acc_loss = 0.0
        acc_norm = 1.0 / self.acc_rate
        total = len(data_loader.dataset)
        batch_size = data_loader.batch_size

        t = tqdm.tqdm(enumerate(data_loader), total=int(math.ceil(total / batch_size)))
        self.optimizer.zero_grad()

        for i, batch in t:
            batch = batch.to(self.device)
            batch_target = batch.y.to(self.device)
            batch_params = batch.params.to(self.device)
            #print(f"Model params : {batch_params}")
            #print(f"Model target : {batch_target}")

            if self.loss_func == F.binary_cross_entropy:
                weights_real = batch_target * self._category_weights[1]
                weights_fake = (1 - batch_target) * self._category_weights[0]
                cat_weights = weights_real + weights_fake
            else:
                cat_weights = self._category_weights

            # Forward
            if self.parallel:
                batch_output = self.model(batch.to_data_list())
            else :
                batch_output = self.model(batch)

            batch_output = batch_output.to(self.device)
            #batch_output = self.model(batch)
            #batch_loss = acc_norm * self.loss_func(batch_output, batch_target, weight=cat_weights)
            batch_loss = acc_norm * self.loss_func(batch_output, batch_target)
            #print(f"Model output : {batch_output}")


            batch_loss.backward()
            """
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    print(name, param.grad.abs().mean().item())
            """
            acc_loss += batch_loss.item()
            sum_loss += batch_loss.item()

            if (i + 1) % self.acc_rate == 0 or (i + 1) == len(data_loader):
                self.optimizer.step()
                self.optimizer.zero_grad()
                t.set_description(f"loss = {acc_loss:.5f}")
                summary['acc_loss'].append(acc_loss)
                summary['acc_lr'].append(self.optimizer.param_groups[0]['lr'])
                if self.lr_sched == 'TorchCyclic':
                    self.lr_scheduler.step()
                acc_loss = 0.0

        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_time'] = time.time()
        summary['train_loss'] = self.acc_rate * sum_loss / (i + 1)
        self.logger(f'Training loss: {summary["train_loss"]:.5f}')
        if '{}'.format(self.device) != 'cpu':
            self.logger(f'Max memory usage: {torch.cuda.max_memory_allocated(self.device)}')
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        summary = dict()
        sum_loss = 0

        total = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        t = tqdm.tqdm(enumerate(data_loader), total=int(math.ceil(total / batch_size)))

        for i, batch in t:
            batch = batch.to(self.device)
            batch_target = batch.y
            batch_params = batch.params.to(self.device)

            #batch_output = self.model(batch, batch_params)
            
            if self.parallel:
                batch_output = self.model(batch.to_data_list())
            else :
                batch_output = self.model(batch)
            #print(f"Model output : {batch_output}")
            #print(f"Model target : {batch_target}")
            #if self.loss_func == F.binary_cross_entropy:
                #weights_real = batch_target * self._category_weights[1]
                #weights_fake = (1 - batch_target) * self._category_weights[0]
                #cat_weights = weights_real + weights_fake
            #else:
                #cat_weights = self._category_weights
            
       
            #batch_output = self.model(batch)
            batch_loss = self.loss_func(batch_output, batch_target)
            sum_loss += batch_loss.item()

        summary['valid_time'] = time.time()
        summary['valid_loss'] = sum_loss / (i + 1)
        summary['valid_acc'] = 0  # Placeholder if you want to add accuracy later

        self.logger(f'Validation loss: {summary["valid_loss"]:.5f}')
        return summary

@torch.no_grad()
def run_inference(model, data_loader, device):
    print(model.module.drn.datanorm.shape)
    model.eval()
    predM = []
    predClass = []

    for batch in data_loader:
        batch = batch.to(device)
        #print(batch[0].x.shape)
        batch_for_drn = batch.clone()
        # Forward pass through the full model
        output = model(batch)  # final classifier output
        drn_out = model.module.drn(batch_for_drn)  # intermediate DRN output
    
        predM.append(drn_out.cpu())
        predClass.append(output.cpu())
    
    return torch.cat(predM), torch.cat(predClass)
