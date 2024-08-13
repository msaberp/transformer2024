import hydra
from omegaconf import DictConfig
import torch
from util.logger import get_logger
from train import run

logger = get_logger(__name__)


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info(cfg)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    cfg.runner.device = device

    run(cfg)

if __name__ == "__main__":
    main()
