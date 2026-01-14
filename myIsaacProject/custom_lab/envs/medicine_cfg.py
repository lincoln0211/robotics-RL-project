import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

USD_PATH_B = PROJECT_ROOT / "assets" / "medicine" / "byung_S.usd"
assert USD_PATH_B.exists(), f"USD not found: {USD_PATH_B}"

USD_PATH_M = PROJECT_ROOT / "assets" / "medicine" / "cold_S.usd"
assert USD_PATH_M.exists(), f"USD not found: {USD_PATH_M}"

USD_PATH_T = PROJECT_ROOT / "assets" / "medicine" / "kitty_S.usd"
assert USD_PATH_T.exists(), f"USD not found: {USD_PATH_T}"


bacchus_cfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/bacchus", spawn=sim_utils.UsdFileCfg(usd_path=str(USD_PATH_B)), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.72)))
cold_cfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/cold", spawn=sim_utils.UsdFileCfg(usd_path=str(USD_PATH_M)), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.15, 0.72)))
kitty_cfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/kitty", spawn=sim_utils.UsdFileCfg(usd_path=str(USD_PATH_T)), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.15, 0.72)))











