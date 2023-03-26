import argparse
import os

from networks.trainers.clcl_trainer import CLCLSA_Trainer


if __name__ == '__main__':
    # python main_clcl.py --data_folder=ROSMAP --hidden_dim=300 --num_epoch=2500
    # python main_clcl.py --data_folder=BRCA --hidden_dim=200 --num_epoch=2500
    parser = argparse.ArgumentParser()

    # dataset settings
    parser.add_argument('--data_folder', type=str, default="LGG")
    parser.add_argument('--missing_rate', type=float, default=0.2)
    parser.add_argument('--exp', type=str, default="./exp")

    # model params
    parser.add_argument('--hidden_dim', type=str, default="300")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--prediction', type=str, default="64,32")
    parser.add_argument('--device', type=str, default="cuda")

    parser.add_argument('--lambda_cl', type=float, default=0.05)
    parser.add_argument('--lambda_co', type=float, default=0.02)

    parser.add_argument('--lambda_al', type=float, default=1.)

    # training params
    parser.add_argument('--num_epoch', type=int, default=2500)
    parser.add_argument('--test_inverval', type=int, default=50)

    args = parser.parse_args()
    params = vars(args)
    params['hidden_dim'] = [int(x) for x in params['hidden_dim'].split(",")]
    params['prediction'] = {i: [int(x) for x in params['prediction'].split(",")] for i in range(3)}
    cl_trainer = CLCLSA_Trainer(params)
    cl_trainer.train()