from trainer import Trainer
from utils.config import load_config
from utils.utils import seed_all

if __name__ == "__main__":
    opt = load_config()
    seed_all(opt.seed)
    trainer = Trainer(opt)
    trainer.train()
