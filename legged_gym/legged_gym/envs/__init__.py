
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from legged_gym.utils.task_registry import task_registry

from .h1.h1_jumpjack_config import H1JumpJackCfg, H1JumpJackCfgPPO
from .base.h1_jumpjack import H1JumpJack
task_registry.register( "h1:jumpjack", H1JumpJack, H1JumpJackCfg(), H1JumpJackCfgPPO())

from .g1.g1_jumpjack_config import G1JumpJackCfg, G1JumpJackCfgPPO
from .base.g1_jumpjack import G1JumpJack
task_registry.register( "g1:jumpjack", G1JumpJack, G1JumpJackCfg(), G1JumpJackCfgPPO())