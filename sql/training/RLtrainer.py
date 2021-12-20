import time

class Trainer:
    
    def __init__(self, model, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # print('m1', self.model)

    def train(self, n_eps):
        pass

class OfflineTrainer(Trainer):

    def __init__(self, model, optimizer, scheduler, batch_sampler, batch_size=256, sequence_length=20, per_batch=True, grad_norm_clip=0.25):
        '''
        Inputs:
        batch_sampler(Z, K): function that samples batches of size Z with sequence length K from the dataset
        '''
        # print('mm', model)
        # print(scheduler)
        Trainer.__init__(self, model, optimizer, scheduler=scheduler)
        self.batch_sampler=batch_sampler
        self.batch_size=batch_size
        self.sequence_length=sequence_length
        self.per_batch=per_batch
        self.grad_norm_clip = grad_norm_clip

    def train(self, n_batches, **kwargs):
        start_time = time.time()
        # print('sm', self.model)
        self.model.train()
        self._train(n_batches, **kwargs)
        training_time = time.time()-start_time
        print('Training time: ', training_time)
        print('Training time per batch: ', training_time/n_batches)

    def _train(self, n_batches, **kwargs):
        pass