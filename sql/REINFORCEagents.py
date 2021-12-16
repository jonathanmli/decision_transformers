class ReinforceAgent(Agent):
    def __init__(self, env, gamma=1, hidden_dim=64, lr=0.001, alpha=1):
        Agent.__init__(self, env)
        self.model = LogitsNet(self.input_dim, hidden_dim, self.output_dim)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.policy_losses, self.value_losses, self.entropy_losses = [], [], []
        self.alpha = alpha
        pass


    def select_action(self, state):
        # print(state)
        logits = self.model(torch.FloatTensor(state))
        pi = Categorical(logits=logits)
        action = pi.sample()
        log_prob = pi.log_prob(action)
        action = action.item()
        return action, log_prob

    # returns tensor (always use tensors) of discounted rewards G_t
    def discounted_rewards(self, rewards):
        discounted_rewards = []
        for t in range(len(rewards)):
            G_t = 0
            pw = 0

            for r in rewards[t:]:
                G_t = G_t + self.gamma ** pw * r
                pw += 1

            discounted_rewards.append(G_t)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        return discounted_rewards

    def deltas(self, rewards, states):
        return self.discounted_rewards(rewards)

    def update(self, logp, rewards, states):
        discounted_rewards = self.deltas(rewards, states)

        #negative because log prob is always negative
        loss = - self.alpha * torch.stack(logp) @ discounted_rewards

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # record for reference later
        self.policy_losses.append(loss.item())

    def train(self, n_episodes=1000):
        for episode in range(n_episodes):

            done = False
            s = self.observe(self.env.reset())
            logp = []
            # actions = []
            rewards = []
            states = []

            #generate episode
            while not done:

                action, log_prob = self.select_action(s)
                s_prime, reward, done, _ = self.env.step(action)
                s_prime = self.observe(s_prime)
                logp.append(log_prob)
                rewards.append(reward)
                states.append(s)

                s = s_prime
                # if episode % 100 == 0:
                #     env.render()

            self.all_rewards.append(np.sum(rewards))
            if episode % 100 == 0:
                print(f'Episode {episode} Score: {np.sum(rewards)}')

            #update models
            self.update(logp, rewards, states)
            self.n_episodes += 1

    def plot_policy_losses(self, window=50, only_mean=False, label='REINFORCE'):
        if not only_mean:
            plt.plot(self.policy_losses, label=label+' raw')
        plt.plot([np.mean(self.policy_losses[i:i + window]) for i in range(self.n_episodes - window)], label=label+' rolling mean')





class ReinforceBaselineAgent(ReinforceAgent):
    def __init__(self, env, gamma=1, hidden_dim=64, lr=0.001, alpha=1, v_alpha=1):
        ReinforceAgent.__init__(self, env, gamma=gamma, hidden_dim=hidden_dim, lr=lr, alpha=alpha)
        # add state-value model
        self.sv_model = LogitsNet(self.input_dim, hidden_dim, 1)
        self.sv_optimizer = Adam(self.sv_model.parameters(), lr=lr)
        self.v_alpha = v_alpha

    def gv_difference(self, rewards, states):
        discounted_rewards = self.discounted_rewards(rewards)
        values = torch.stack([torch.squeeze(self.sv_model(torch.FloatTensor(_))) for _ in states])
        deltas = discounted_rewards - values

        # difference should only link back to sv model
        return deltas

    def deltas(self, rewards, states):
        return self.gv_difference(rewards, states)

    def v_loss(self, rewards, states):

        # mse between targets and current
        return (self.gv_difference(rewards, states) ** 2).mean()

    def update(self, logp, rewards, states):
        deltas = self.deltas(rewards, states)

        #make sure to detach graph from deltas so our loss does not go to value function
        loss = - self.alpha * torch.stack(logp) @ deltas.detach()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #backpropogate for value loss
        v_loss = self.v_alpha * self.v_loss(rewards, states)
        self.sv_optimizer.zero_grad()
        v_loss.backward()
        self.sv_optimizer.step()

        # record for reference later
        self.policy_losses.append(loss.item())
        self.value_losses.append(v_loss.item())

    def plot_value_losses(self, window=50, only_mean=False, label='REINFORCE'):
        if not only_mean:
            plt.plot(self.value_losses, label=label+' raw')
        plt.plot([np.mean(self.value_losses[i:i + window]) for i in range(self.n_episodes - window)], label=label+' rolling mean')


class ReinforceBaselineTDAgent(ReinforceBaselineAgent):

    def v_loss(self, rewards, states):
        values = [torch.squeeze(self.sv_model(torch.FloatTensor(_))) for _ in states]
        losses = []
        for t in range(len(rewards)):
            v = values[t]
            if t + 1 < len(rewards):
                v_p = values[t+1]
            else:
                v_p = 0
            r = rewards[t]
            losses.append(r + self.gamma * v_p - v)

        return ((torch.stack(losses)) ** 2).mean()