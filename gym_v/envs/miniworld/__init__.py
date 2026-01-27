"""MiniWorld 3D navigation environments for VLM training."""

from gym_v.envs.miniworld.base import MiniWorldBaseEnv
from gym_v.envs.miniworld.collecthealth import MiniWorldCollectHealthEnv
from gym_v.envs.miniworld.hallway import MiniWorldHallwayEnv
from gym_v.envs.miniworld.maze import (
    MiniWorldMazeEnv,
    MiniWorldMazeS2Env,
    MiniWorldMazeS3Env,
    MiniWorldMazeS3FastEnv,
)
from gym_v.envs.miniworld.oneroom import (
    MiniWorldOneRoomEnv,
    MiniWorldOneRoomS6Env,
    MiniWorldOneRoomS6FastEnv,
)
from gym_v.envs.miniworld.pickup import MiniWorldPickupObjectsEnv
from gym_v.envs.miniworld.putnext import MiniWorldPutNextEnv
from gym_v.envs.miniworld.rooms import (
    MiniWorldFourRoomsEnv,
    MiniWorldRoomObjectsEnv,
    MiniWorldThreeRoomsEnv,
)
from gym_v.envs.miniworld.sidewalk import MiniWorldSidewalkEnv
from gym_v.envs.miniworld.sign import MiniWorldSignEnv
from gym_v.envs.miniworld.tmaze import (
    MiniWorldTMazeEnv,
    MiniWorldTMazeLeftEnv,
    MiniWorldTMazeRightEnv,
)
from gym_v.envs.miniworld.wallgap import MiniWorldWallGapEnv
from gym_v.envs.miniworld.ymaze import (
    MiniWorldYMazeEnv,
    MiniWorldYMazeLeftEnv,
    MiniWorldYMazeRightEnv,
)
