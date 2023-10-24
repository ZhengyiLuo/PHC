import math


class LoggerRL:

    def __init__(self):
        self.num_steps = 0
        self.num_episodes = 0
        self.avg_episode_len = 0
        self.total_reward = 0
        self.min_episode_reward = math.inf
        self.max_episode_reward = -math.inf
        self.total_c_reward = 0
        self.min_c_reward = math.inf
        self.max_c_reward = -math.inf
        self.episode_reward = 0
        self.avg_episode_reward = 0
        self.avg_c_reward = 0
        self.avg_episode_c_reward = 0
        self.total_c_info = 0
        self.avg_c_info = 0
        self.avg_episode_c_info = 0
        self.sample_time = 0

    def start_episode(self, env):
        self.episode_reward = 0

    def step(self, env, reward, c_reward, c_info, info):
        self.episode_reward += reward
        self.total_c_reward += c_reward
        self.total_c_info += c_info
        self.min_c_reward = min(self.min_c_reward, c_reward)
        self.max_c_reward = max(self.max_c_reward, c_reward)
        self.num_steps += 1

    def end_episode(self, env):
        self.num_episodes += 1
        self.total_reward += self.episode_reward
        self.min_episode_reward = min(self.min_episode_reward, self.episode_reward)
        self.max_episode_reward = max(self.max_episode_reward, self.episode_reward)

    def end_sampling(self):
        self.avg_episode_len = self.num_steps / self.num_episodes
        self.avg_episode_reward = self.total_reward / self.num_episodes
        self.avg_c_reward = self.total_c_reward / self.num_steps
        self.avg_c_info = self.total_c_info / self.num_steps
        self.avg_episode_c_reward = self.total_c_reward / self.num_episodes
        self.avg_episode_c_info = self.total_c_info / self.num_episodes

    @classmethod
    def merge(cls, logger_list):
        logger = cls()
        logger.total_reward = sum([x.total_reward for x in logger_list])
        logger.num_episodes = sum([x.num_episodes for x in logger_list])
        logger.num_steps = sum([x.num_steps for x in logger_list])
        logger.avg_episode_len = logger.num_steps / logger.num_episodes
        logger.avg_episode_reward = logger.total_reward / logger.num_episodes
        logger.max_episode_reward = max([x.max_episode_reward for x in logger_list])
        logger.min_episode_reward = max([x.min_episode_reward for x in logger_list])
        logger.total_c_reward = sum([x.total_c_reward for x in logger_list])
        logger.avg_c_reward = logger.total_c_reward / logger.num_steps
        logger.max_c_reward = max([x.max_c_reward for x in logger_list])
        logger.min_c_reward = min([x.min_c_reward for x in logger_list])
        logger.total_c_info = sum([x.total_c_info for x in logger_list])
        logger.avg_c_info = logger.total_c_info / logger.num_steps
        logger.avg_episode_c_reward = logger.total_c_reward / logger.num_episodes
        logger.avg_episode_c_info = logger.total_c_info / logger.num_episodes
        return logger