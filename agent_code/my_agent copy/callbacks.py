import logging
import numpy as np
import os
import torch
from .features import state_to_features 
from .train import PPO, ACTIONS, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP  # 导入超参数
import events as e

def save_model(self):
    torch.save(self.model.state_dict(), "my-saved-model.pt")

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that act(...) can be called.
    """
    # 初始化 PPO 模型（训练模式或加载已有模型）
    def initialize_model():
        return PPO(
            state_dim=588,
            action_dim=len(ACTIONS),
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            gamma=GAMMA,
            k_epochs=K_EPOCHS,
            eps_clip=EPS_CLIP
        )

    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = initialize_model()
    else:
        self.logger.info("Loading model from saved state.")
        self.model = initialize_model()
        self.model.load_state_dict(torch.load("my-saved-model.pt", weights_only=True))  # 加载模型参数

    # 初始化事件统计字典
    self.event_counts = {
        'STAYED_TOO_LONG': 0,
        'CLOSER_TO_COIN': 0,
        'FURTHER_FROM_COIN': 0,
        'OLD_AREA_REVISITED': 0,
        'NEW_AREA_EXPLORED': 0,
        'MOVED_AWAY_FROM_BOMB': 0,
        'IN_EXPLOSION_RANGE': 0,
        e.INVALID_ACTION: 0,
        e.COIN_COLLECTED: 0,
        e.KILLED_SELF: 0,
        e.BOMB_DROPPED: 0,
        e.GOT_KILLED: 0,
        e.SURVIVED_ROUND: 0,
        e.CRATE_DESTROYED: 0
    }

    # 配置日志记录器
    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(__name__)

    # 初始化位置、炸弹相关和动作记录
    self.last_positions = []
    self.action_sources = []
    self.coverage_map = np.zeros((17, 17))
    self.bomb_placed = False
    self.bomb_position = None
    self.bomb_place_position = None

    # 绑定 _save_action_sources 方法到 self 对象
    self._save_action_sources = lambda: save_action_sources(self)

def save_action_sources(self):
    with open("action_sources.txt", "a") as f:
        f.write(f"{self.action_sources[-1]}\n")

def act(self, game_state: dict) -> str:
    # 在每个回合的第一个时间步进行初始化
    if game_state['step'] == 1:
        self.bomb_cooldown = 10  # 炸弹冷却时间，单位为步数
        self.last_bomb_time = -self.bomb_cooldown  # 初始化冷却时间
        self.logger.info("回合开始，初始化冷却时间")
    
    # 获取当前代理的位置
    self_x, self_y = game_state['self'][3]
    arena = game_state['field'].copy()  # 拷贝一份arena，防止直接修改
    current_step = game_state['step']  # 获取当前步数

    # 冷却时间优先判断：如果处于冷却状态，禁用放炸弹和等待，但继续进行后续判断
    if current_step - self.last_bomb_time < self.bomb_cooldown:
        print(current_step,self.last_bomb_time,self.bomb_cooldown)
        print("炸弹冷却中，禁用炸弹和等待")
        action_probs = self.model.actor(torch.tensor(state_to_features(game_state), dtype=torch.float32).unsqueeze(0)).detach().numpy().flatten()
        action_probs[ACTIONS.index('BOMB')] = 0  # 禁用放炸弹的动作
        action_probs[ACTIONS.index('WAIT')] = 0  # 禁用等待的动作
        action_probs /= action_probs.sum()  # 重新归一化概率
        
        # 选择其他动作继续后续规则
        action = np.random.choice(ACTIONS, p=action_probs)
        print(f"冷却中，选择其他动作: {action}")
    else:
        # 检查代理是否连续待在同一个位置
        if self.last_positions and self.last_positions[-1] == (self_x, self_y):
            self.stay_count += 1
        else:
            self.stay_count = 0
        self.last_positions.append((self_x, self_y))

        # 保持最近5步的位置记录
        if len(self.last_positions) > 5:
            self.last_positions.pop(0)

        # 检查代理身边是否有箱子或敌人（仅在不处于冷却状态下才检查）
        crate_nearby = any(
            arena[self_x + dx, self_y + dy] not in [0, -2]   # 检查上下左右是否有箱子
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        )
        others = game_state['others']
        enemy_nearby = any(
            abs(self_x - x) + abs(self_y - y) <= 1
            for _, _, _, (x, y) in others
        )

        should_place_bomb = crate_nearby or enemy_nearby

        # 设置炸弹放置的阈值
        bomb_threshold = 0.3

        # 如果应该放炸弹，检查是否能在2步内找到安全逃生方向
        if should_place_bomb:
            # 模拟放置炸弹，计算炸弹爆炸范围
            bomb_position = (self_x, self_y)
            explosion_range = calculate_explosion_range(bomb_position, arena)
            print(f"炸弹爆炸范围: {explosion_range}")

            # 在 arena 上标记爆炸范围
            for (ex, ey) in explosion_range:
                arena[ex, ey] = -2  # 使用 -2 表示爆炸范围

            # 检查当前位置的四个相邻方向
            def can_escape(x, y):
                # 获取四个方向的相邻格子
                neighbors = [(x + dx, y + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
                # 过滤出可移动的格子（即值为0或-2的格子）
                safe_directions = [(nx, ny) for nx, ny in neighbors if arena[nx, ny] in [0, -2]]

                # 检查以下几种情况：
                # 1. 两个相邻格子都为0
                # 2. 第一步为-2，第二步为0
                for direction in safe_directions:
                    nx, ny = direction
                    # 检查第二步
                    next_neighbors = [(nx + dx, ny + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
                    for next_nx, next_ny in next_neighbors:
                        if arena[nx, ny] == -2 and arena[next_nx, next_ny] == 0:
                            return True
                        elif arena[nx, ny] == 0 and arena[next_nx, next_ny] == 0:
                            return True

                return False

            # 判断当前所在位置是否能在2步内找到安全逃生方向
            if can_escape(self_x, self_y):
                # 获取放炸弹的概率
                action_probs = self.model.actor(torch.tensor(state_to_features(game_state), dtype=torch.float32).unsqueeze(0)).detach().numpy().flatten()
                bomb_prob = action_probs[ACTIONS.index('BOMB')]
                
                # 检查概率是否低于阈值
                if bomb_prob < bomb_threshold:
                    print("放炸弹的概率低于阈值，禁用炸弹和等待")
                    action_probs[ACTIONS.index('BOMB')] = 0  # 禁用放炸弹的动作
                    action_probs[ACTIONS.index('WAIT')] = 0  # 禁用等待的动作
                    action_probs /= action_probs.sum()  # 重新归一化概率
                    action = np.random.choice(ACTIONS, p=action_probs)
                    print(f"选择其他动作: {action}")
                    return action

                print("可以在2步内逃生，放置炸弹")
                self.last_bomb_time = current_step  # 记录放炸弹的时间
                self.bomb_place_position = (self_x, self_y)  # 记录炸弹放置的位置
                return 'BOMB'
            else:
                print("无法在2步内逃生，禁用炸弹和等待")
                # 禁用炸弹和等待动作，重新选择其他动作
                action_probs = self.model.actor(torch.tensor(state_to_features(game_state), dtype=torch.float32).unsqueeze(0)).detach().numpy().flatten()
                action_probs[ACTIONS.index('BOMB')] = 0  # 禁用放炸弹的动作
                action_probs[ACTIONS.index('WAIT')] = 0  # 禁用等待的动作
                action_probs /= action_probs.sum()  # 重新归一化概率
                action = np.random.choice(ACTIONS, p=action_probs)
                print(f"无法逃生，选择其他动作: {action}")
        else:
            # 没有炸弹放置条件，直接选择动作
            action_probs = self.model.actor(torch.tensor(state_to_features(game_state), dtype=torch.float32).unsqueeze(0)).detach().numpy().flatten()
            action_probs[ACTIONS.index('BOMB')] = 0  # 禁用放炸弹的动作
            action_probs[ACTIONS.index('WAIT')] = 0  # 禁用等待的动作
            action_probs /= action_probs.sum() 
            action = np.random.choice(ACTIONS, p=action_probs)
            print(f"没有炸弹条件，选择其他动作: {action}")

    # 执行动作并检查位置变化
    new_x, new_y = self_x, self_y  # 假设执行动作后的位置
    if action == 'UP':
        new_y += 1
    elif action == 'DOWN':
        new_y -= 1
    elif action == 'LEFT':
        new_x -= 1
    elif action == 'RIGHT':
        new_x += 1

    # 检查目标位置是否可移动
    if arena[new_x, new_y] != 0:  # 0 表示可移动的空地
        print("检查可选方向的地图：",arena)
        random_prob = 0.5  # 50% 概率随机选择动作
        if np.random.rand() < random_prob:
            available_actions = [a for a in ACTIONS if 
                        (a == 'DOWN' and arena[self_x, self_y + 1] == 0) or
                        (a == 'UP' and arena[self_x, self_y - 1] == 0) or
                        (a == 'LEFT' and arena[self_x - 1, self_y] == 0) or
                        (a == 'RIGHT' and arena[self_x + 1, self_y] == 0)]
            print("可选动作如下:", available_actions)
            if available_actions:
                action = np.random.choice(available_actions)
                self.action_sources.append("Random for invalid action")
                self._save_action_sources()
                print("出现无效动作，选择动作：", action)

    # 检查代理是否在循环模式
    loop_detected = False
    if len(self.last_positions) >= 4:
        loop_patterns = [
            [(self_x, self_y), (self_x + 1, self_y), (self_x, self_y), (self_x - 1, self_y)],
            [(self_x, self_y), (self_x - 1, self_y), (self_x, self_y), (self_x + 1, self_y)],
            [(self_x, self_y), (self_x, self_y + 1), (self_x, self_y), (self_x, self_y - 1)],
            [(self_x, self_y), (self_x, self_y - 1), (self_x, self_y), (self_x, self_y + 1)]
        ]
        if self.last_positions[-4:] in loop_patterns:
            loop_detected = True
            self.logger.debug("Detected looping pattern.")

    # 如果检测到循环模式或连续待在同一个位置，则按指定概率随机选择一个动作
    if loop_detected or self.stay_count > 3:
        random_prob = 0.5  # 50% 概率选择
        if np.random.rand() < random_prob:
            self.logger.debug("Choosing action purely at random due to loop or staying too long.")
            action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25, .0, .0])
            print("随机选择动作 因为 loop or staying")
            self._save_action_sources()
            return action

    

    # 返回最终选择的动作
    random_prob = 0.1  # 探索的概率
    if self.train and np.random.rand() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[0.25, 0.25, 0.25, 0.25, 0, 0])
    self.logger.info(f"Selected action: {action}")
    self.action_sources.append("Model")
    self._save_action_sources()
    return action

def calculate_explosion_range(bomb_position, arena, explosion_radius=3):
    """
    计算炸弹的爆炸范围，考虑墙壁和障碍物。
    
    :param bomb_position: 炸弹位置 (x, y)
    :param arena: 游戏场景 (二维数组，0表示可通行，-1表示墙壁)
    :param explosion_radius: 爆炸半径（向四个方向的格子数）
    :return: 爆炸范围的坐标集合
    """
    x, y = bomb_position
    explosion_range = set()

    # 加入炸弹位置
    explosion_range.add((x, y))

    # 向左扩展
    for i in range(1, explosion_radius + 1):
        if arena[x - i, y] == -1:  # 遇到墙壁或障碍物
            break
        explosion_range.add((x - i, y))
    
    # 向右扩展
    for i in range(1, explosion_radius + 1):
        if arena[x + i, y] == -1:
            break
        explosion_range.add((x + i, y))
    
    # 向上扩展
    for i in range(1, explosion_radius + 1):
        if arena[x, y - i] == -1:
            break
        explosion_range.add((x, y - i))
    
    # 向下扩展
    for i in range(1, explosion_radius + 1):
        if arena[x, y + i] == -1:
            break
        explosion_range.add((x, y + i))

    return explosion_range