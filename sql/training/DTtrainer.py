from training.RLtrainer import OfflineTrainer
import torch

class DT_Trainer(OfflineTrainer):
    # def __init__(self, model, optimizer, scheduler, batch_sampler):
    #     super().__init__(model, optimizer, scheduler, batch_sampler,)

    def _train(self, n_batches):
        # train DT agent
        for i in range(n_batches):
            s, a, r, d, rtg, timesteps, mask = self.batch_sampler(self.batch_size)
            self._train_on_batch(s, a, r, d, rtg, timesteps, mask, per_batch=self.per_batch)
            # print('Batch number', i)
        
        pass

    def _train_on_batch(self, states, actions, rewards, dones, rtg, timesteps, mask, per_batch=True):
        """
        Input: tensors of states of dimension Z * K * X, where Z is number of trajectories (batch size), K is trajectory
                length, and X is the dimension of the states
                similar for rewards, ... etc

                Note that the rtgs inputs may have trajectory lengths +1 for padding
                The rtgs are normalized already

        Trains the agent
        """
        if per_batch:
            a_preds = self.model(rtg[:, :-1], states, actions, timesteps, mask)

            # this loss is weird -- why r we taking diff in actions?
            loss = torch.mean((a_preds - actions) ** 2)
#             print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
#                 print('sss')
        else:
            # note that we optimize over each batch instead of each data here. the paper optimized per data
            # loop over each batch of size K
            for i in range(len(states)):
                # print(len(states[i]))
                # print(len(rtg[i]))
                # print(len(actions[i]))
                # print(len(timesteps[i]))

                a_preds = self.model(rtg[i][:-1], states[i], actions[i], timesteps[i], mask[i])

#                 print('ap', a_preds)
#                 print('aa', actions[i])

                # # this loss is weird -- why r we taking diff in actions?
                loss = torch.mean((a_preds - actions[i]) ** 2)
#                 print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()