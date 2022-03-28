"""Spinning Up algorithms ported to TensorFlow v2."""

from .algos.tf2.ddpg.ddpg import ddpg  # pylint: disable=wrong-import-position
from .algos.tf2.ppo.ppo import ppo  # pylint: disable=wrong-import-position
from .algos.tf2.sac.sac import sac  # pylint: disable=wrong-import-position
from .algos.tf2.td3.td3 import td3  # pylint: disable=wrong-import-position
from .algos.tf2.vpg.vpg import vpg  # pylint: disable=wrong-import-position
from .algos.tf2.sop.sop import sop  # pylint: disable=wrong-import-position


ddpg_tf2 = ddpg
ppo_tf2 = ppo
sac_tf2 = sac
td3_tf2 = td3
vpg_tf2 = vpg
sop_tf2 = sop
