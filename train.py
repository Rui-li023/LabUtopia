"""
Usage:
Training:
python train.py --config-name=train_act_image_workspace
"""

import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), mode='w', buffering=1, closefd=False)
sys.stderr = os.fdopen(sys.stderr.fileno(), mode='w', buffering=1, closefd=False)

import hydra
from omegaconf import OmegaConf
import pathlib
from policy.workspace.base_workspace import BaseWorkspace

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'policy','config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
