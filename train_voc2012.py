import torch
import yaml

from action.train_voc import train

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    with open(f"config/plan/{input('plan: ')}.yml", 'r') as fd:
        train(yaml.safe_load(fd), device=torch.device('cuda'))
