import argparse
import os
import pandas as pd
import pprint

from networks.trainers.clcl_trainer import CLUECL_Trainer
from sklearn.model_selection import ParameterGrid


def is_exist(target_csv_path, params):
    if not os.path.isfile(target_csv_path):
        return False
    else:
        df = pd.read_csv(target_csv_path, sep='\t')
        if df[(df['lambda_cil']==params['lambda_cil']) & 
            (df['lambda_ccl']==params['lambda_ccl']) & 
            (df['lambda_co']==params['lambda_co']) & 
            (df['lambda_al']==params['lambda_al']) & 
            (df['dropout']==params['dropout']) & 
            (df['missing_rate']==params['missing_rate'])].shape[0]>0:
            return True
        else:
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset settings
    parser.add_argument('--data_folder', type=str, default="KIPAN")
    #parser.add_argument('--missing_rate', type=float, default=0.2)
    parser.add_argument('--exp', type=str, default="./exp_tmp")

    # model params
    parser.add_argument('--hidden_dim', type=str, default="200") # ROSMAP 300, OTHER 200
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--prediction', type=str, default="64,32")
    parser.add_argument('--device', type=str, default="0")

    # parser.add_argument('--lambda_cil', type=float, default=0.05) # constrastive instance
    # parser.add_argument('--lambda_ccl', type=float, default=0.05) # constrastive cluster
    # parser.add_argument('--lambda_co', type=float, default=0.02)  # cross omics
    # parser.add_argument('--lambda_al', type=float, default=1.)

    # training params
    parser.add_argument('--num_epoch', type=int, default=2500)
    parser.add_argument('--test_inverval', type=int, default=50)

    args = parser.parse_args()

    exp = args.exp
    dataset = args.data_folder


    # set device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"  #

    params = vars(args)
    params['device'] = "cuda"
    for k in params.keys():
        params[k] = [params[k]]
    # params['hidden_dim'] = [[300], [200]]
    # params['prediction'] = [{i: [[64, 32]] for i in range(3)}, {i: [[128, 64]] for i in range(3)}]

    #params['lambda_cil'] = [.0, 0.004, 0.005, 0.01]
    params['lambda_cil'] = [.0, 0.01, 0.02, 0.03, 0.04, 0.05][::-1]
    params['lambda_ccl'] = [.0]
    params['lambda_co'] = [.0, 0.01, 0.02, 0.03, 0.04, 0.05][::-1]
    params['lambda_al'] = [.0, 0.01, 0.02, 0.03, 0.04, 0.05][::-1]
    params['dropout'] = [.5]
    params['missing_rate'] = [.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][::-1]


    # grid search
    total = len(list(ParameterGrid(params)))

    if not os.path.isdir(exp):
        os.makedirs(exp)
    
    if os.path.isfile(os.path.join(exp, f"best_results_{dataset}.csv")):
        n_exp_have_done = len(open(os.path.join(exp, f"best_results_{dataset}.csv"), "r").readlines())-1
        target = open(os.path.join(exp, f"best_results_{dataset}.csv"), "a")
    else:
        n_exp_have_done = 0
        target = open(os.path.join(exp, f"best_results_{dataset}.csv"), "w")
        if dataset in ["ROSMAP", "LGG", "U19B"]: # binary classification
            target.write("lambda_cil\tlambda_ccl\tlambda_co\tlambda_al\tdropout\tmissing_rate\tacc\tf1\tauc\tpath\n")
        elif dataset in ["BRCA", "KIPAN", "U19M"]: # multi class
            target.write("lambda_cil\tlambda_ccl\tlambda_co\tlambda_al\tdropout\tmissing_rate\tacc\tf1w\tf1m\tpath\n")
        target.flush()

    c = 0
    for parameters_, parameters_id in zip(list(ParameterGrid(params)), range(len(list(ParameterGrid(params))))):
        # if exist, continue
        if is_exist(os.path.join(exp, f"best_results_{dataset}.csv"), parameters_):
            c += 1
            continue
        # adjust grid
        if parameters_['missing_rate'] > 0. and parameters_['lambda_co'] <= 0.: # must train cross embedding if using missing data
            continue
        
        if parameters_['lambda_co'] > 0. and parameters_['missing_rate'] == 0.:
            continue

        dicts = []
        print(f"--------------{c}/{total}---------")
        pprint.pprint(parameters_)
        parameters_['hidden_dim'] = [int(x) for x in parameters_['hidden_dim'].split(",")]
        parameters_['prediction'] = {i: [int(x) for x in parameters_['prediction'].split(",")] for i in range(3)}
        cl_trainer = CLUECL_Trainer(parameters_)
        best_dict, exp_path = cl_trainer.train()
        
        target.write(f"{parameters_['lambda_cil']}\t" \
                     f"{parameters_['lambda_ccl']}\t" \
                     f"{parameters_['lambda_co']}\t" \
                     f"{parameters_['lambda_al']}\t" \
                     f"{parameters_['dropout']}\t" \
                     f"{parameters_['missing_rate']}\t" \
                     f"{best_dict[0]}\t{best_dict[1]}\t{best_dict[2]}\t{exp_path}\n")
        target.flush()
        c += 1

    target.close()

   