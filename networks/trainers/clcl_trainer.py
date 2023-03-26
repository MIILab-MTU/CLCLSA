import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pprint

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils.data_utils import one_hot_tensor, prepare_trte_data, get_mask
from networks.models.clcl import CLUECL3
from datetime import datetime
from tqdm import tqdm


def get_mask_wrapper(n_views, data_len, missing_rate):
    success = False
    while not success:
        try:
            mask = get_mask(n_views, data_len, missing_rate)
            success = True
        except:
            success = False
    
    return mask

class CLCLSA_Trainer(object):

    def __init__(self, params):
        self.params = params
        self.device = self.params['device']
        self.__init_dataset__()
        self.model = CLUECL3(self.dim_list, self.params['hidden_dim'], self.num_class, self.params['dropout'], self.params['prediction'])

    def __init_dataset__(self):
        self.data_tr_list, self.data_test_list, self.trte_idx, self.labels_trte = prepare_trte_data(self.params['data_folder'], True)
        self.labels_tr_tensor = torch.LongTensor(self.labels_trte[self.trte_idx["tr"]])
        num_class = len(np.unique(self.labels_trte))
        self.onehot_labels_tr_tensor = one_hot_tensor(self.labels_tr_tensor, num_class)
        self.labels_tr_tensor = self.labels_tr_tensor.cuda()
        self.onehot_labels_tr_tensor = self.onehot_labels_tr_tensor.cuda()
        dim_list = [x.shape[1] for x in self.data_tr_list]
        self.dim_list = dim_list
        self.num_class = num_class
        print("[x] number of classes = ", self.num_class)

        if self.params['missing_rate'] > 0.:
            mask = get_mask(3, self.data_tr_list[0].shape[0], self.params['missing_rate'])
            mask = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(self.device)
            x1_train = self.data_tr_list[0] * torch.unsqueeze(mask[:, 0], 1)
            x2_train = self.data_tr_list[1] * torch.unsqueeze(mask[:, 1], 1)
            x3_train = self.data_tr_list[2] * torch.unsqueeze(mask[:, 2], 1)
            self.mask_train = mask
            self.data_tr_list = [x1_train, x2_train, x3_train]
            #mask = get_mask(3, self.data_test_list[0].shape[0], self.params['missing_rate'])
            mask = get_mask_wrapper(3, self.data_test_list[0].shape[0], self.params['missing_rate'])
            mask = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(self.device)
            x1_test = self.data_test_list[0] * torch.unsqueeze(mask[:, 0], 1)
            x2_test = self.data_test_list[1] * torch.unsqueeze(mask[:, 1], 1)
            x3_test = self.data_test_list[2] * torch.unsqueeze(mask[:, 2], 1)
            self.mask_test = mask
            self.data_test_list = [x1_test, x2_test, x3_test]


    def train(self):

        exp_name = os.path.join(self.params['exp'], f"{self.params['data_folder']}_{datetime.utcnow().strftime('%B_%d_%Y_%Hh%Mm%Ss')}")
        os.makedirs(exp_name, exist_ok=True)
        with open(os.path.join(exp_name, 'config.json'), 'w') as fp:
            json.dump(self.params, fp, indent=4)

        self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'], weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.params['step_size'], gamma=0.2)
        global_acc = 0.
        best_eval = []
        print("\nTraining...")
        for epoch in tqdm(range(self.params['num_epoch'] + 1)):
            print_loss = True if epoch % self.params['test_inverval'] == 0 else False
            self.train_epoch(print_loss)

            self.scheduler.step()
            if epoch % self.params['test_inverval'] == 0:
                te_prob = self.test_epoch()
                if not np.any(np.isnan(te_prob)):
                    print("\nTest: Epoch {:d}".format(epoch))
                    if self.num_class == 2:
                        acc = accuracy_score(self.labels_trte[self.trte_idx["te"]], te_prob.argmax(1))
                        f1 = f1_score(self.labels_trte[self.trte_idx["te"]], te_prob.argmax(1))
                        auc = roc_auc_score(self.labels_trte[self.trte_idx["te"]], te_prob[:, 1])
                        print(f"Test ACC: {acc:.5f}, F1: {f1:.5f}, AUC: {auc:.5f}")
                        if acc > global_acc:
                            global_acc = acc
                            best_eval = [acc, f1, auc]
                            self.save_checkpoint(exp_name)
                    else:
                        acc = accuracy_score(self.labels_trte[self.trte_idx["te"]], te_prob.argmax(1))
                        f1w = f1_score(self.labels_trte[self.trte_idx["te"]], te_prob.argmax(1), average='weighted')
                        f1m = f1_score(self.labels_trte[self.trte_idx["te"]], te_prob.argmax(1), average='macro')
                        print(f"Test ACC: {acc:.5f}, F1 weighted : {f1w:.5f}, F1 macro: {f1m:.5f}")
                        if acc > global_acc:
                            global_acc = acc
                            best_eval = [acc, f1w, f1m]
                            self.save_checkpoint(exp_name)

        return best_eval, exp_name

    def train_epoch(self, print=False):
        self.model.train()
        self.optimizer.zero_grad()
        if self.params['missing_rate'] > 0:
            loss, _, loss_dict = self.model.train_missing_cg(self.data_tr_list, self.mask_train, self.labels_tr_tensor, self.device,
                aux_loss=self.params['lambda_al']>0, lambda_al=self.params['lambda_al'],
                cross_omics_loss=self.params['lambda_co']>0, lambda_col=self.params['lambda_co'],
                constrastive_loss=self.params['lambda_cl']>0, lambda_cl=self.params['lambda_cl'])
        else:
            loss, _, loss_dict = self.model(self.data_tr_list, self.labels_tr_tensor, 
                                            aux_loss=self.params['lambda_al']>0, lambda_al=self.params['lambda_al'])
        
        if print:
            pprint.pprint(loss_dict)
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            if self.params['missing_rate'] > 0:
                logit = self.model.infer_on_missing(self.data_test_list, self.mask_test, self.device)
            else:
                logit = self.model.infer(self.data_test_list)
            prob = F.softmax(logit, dim=1).data.cpu().numpy()
        return prob

    def save_checkpoint(self, checkpoint_path, filename="checkpoint.pt"):
        os.makedirs(checkpoint_path, exist_ok=True)
        filename = os.path.join(checkpoint_path, filename)
        torch.save(self.model.state_dict(), filename)