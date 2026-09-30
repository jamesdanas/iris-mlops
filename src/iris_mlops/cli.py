import logging  
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from iris_mlops.train import train


# Compute the absolute path to the conf/ directory from this file's location.
# This works no matter where the command is run from.
CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "conf")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # WHY print the config:
    #   Hydra saves it auomatically to outputs/, but printting helps
    #   you see what actually ran.
    print(OmegaConf.to_yaml(cfg))

    acc = train(
        seed=cfg.seed,
        C=cfg.model.C,
        max_iter=cfg.model.max_iter
    )
    print(f"Final accuracy: {acc:4f}")


if __name__ == "__main__":
    main()