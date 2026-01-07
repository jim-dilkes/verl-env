ACTIONS = {
    "right": "move right",
    "down": "move down",
    "left": "move left",
    "up": "move up",
    "stay": "stay in place (wait)",
    "interact": "interact with object in front of you (pick up, place, or use)",
}

ACTION_TO_IDX = {
    "right": 0,
    "down": 1,
    "left": 2,
    "up": 3,
    "stay": 4,
    "interact": 5,
}

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}

DIRECTION_NAMES = {0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT"}
