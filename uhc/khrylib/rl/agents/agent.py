import multiprocessing
import math
import time
import os

from uhc.khrylib.rl.core import LoggerRL, TrajBatch
from uhc.khrylib.utils.memory import Memory
from uhc.khrylib.utils.torch import *
os.environ["OMP_NUM_THREADS"] = "1"


class Agent:

    def __init__(self, env, policy_net, value_net, dtype, device, gamma, data_loader, custom_reward=None,
                 end_reward=True, mean_action=False, render=False, running_state=None, num_threads=1):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.dtype = dtype
        self.device = device
        self.gamma = gamma
        self.custom_reward = custom_reward
        self.end_reward = end_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads
        self.noise_rate = 1.0
        self.traj_cls = TrajBatch
        self.logger_cls = LoggerRL
        self.sample_modules = [policy_net]
        self.update_modules = [policy_net, value_net]
        self.data_loader = data_loader

    def seed_worker(self, pid):
        if pid > 0:
            torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
            np.random.seed(np.random.randint(5000)* pid)
            if hasattr(self.env, 'np_random'):
                self.env.np_random.rand(np.random.randint(5000 )* pid)

    def sample_worker(self, pid, queue, min_batch_size):
        self.seed_worker(pid)

        memory = Memory()
        logger = self.logger_cls()

        while logger.num_steps < min_batch_size:
            self.env.load_expert(self.data_loader.sample_seq())
            
            state = self.env.reset()
            if self.running_state is not None:
                state = self.running_state(state)
            logger.start_episode(self.env)
            self.pre_episode()

            for t in range(10000):
                state_var = tensor(state).unsqueeze(0)
                trans_out = self.trans_policy(state_var)
                mean_action = self.mean_action or self.env.np_random.binomial(1, 1 - self.noise_rate)
                action = self.policy_net.select_action(trans_out, mean_action)[0].numpy()
                action = int(action) if self.policy_net.type == 'discrete' else action.astype(np.float64)
                next_state, env_reward, done, info = self.env.step(action)
                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                # use custom or env reward
                if self.custom_reward is not None:
                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    reward = c_reward
                else:
                    c_reward, c_info = 0.0, np.array([0.0])
                    reward = env_reward
                    
                # add end reward
                if self.end_reward and info.get('end', False):
                    reward += self.env.end_reward
                # logging
                logger.step(self.env, env_reward, c_reward, c_info, info)

                mask = 0 if done else 1
                exp = 1 - mean_action
                self.push_memory(memory, state, action, mask, next_state, reward, exp)

                if pid == 0 and self.render:
                    self.env.render()
                if done:
                    break
                state = next_state

            logger.end_episode(self.env)
        logger.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger])
        else:
            return memory, logger

    def pre_episode(self):
        return

    def push_memory(self, memory, state, action, mask, next_state, reward, exp):
        memory.push(state, action, mask, next_state, reward, exp)

    def pre_sample(self):
        return

    def sample(self, min_batch_size):
        t_start = time.time()
        self.pre_sample()
        to_test(*self.sample_modules) # Sending test modeuls to cpu!!!
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
                queue = multiprocessing.Queue()
                memories = [None] * self.num_threads
                loggers = [None] * self.num_threads
                for i in range(self.num_threads-1):
                    worker_args = (i+1, queue, thread_batch_size)
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                memories[0], loggers[0] = self.sample_worker(0, None, thread_batch_size)

                for i in range(self.num_threads - 1):
                    pid, worker_memory, worker_logger = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger
                traj_batch = self.traj_cls(memories)
                logger = self.logger_cls.merge(loggers)

        logger.sample_time = time.time() - t_start
        return traj_batch, logger

    def trans_policy(self, states):
        """transform states before going into policy net"""
        return states

    def trans_value(self, states):
        """transform states before going into value net"""
        return states

    def set_noise_rate(self, noise_rate):
        self.noise_rate = noise_rate