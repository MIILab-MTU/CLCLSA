import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from networks.models.common_layers import LinearLayer, Prediction
from networks.models.losses import contrastive_Loss

class CLUECL3(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout, prediction_dicts):
        super().__init__()
        assert(len(in_dim)) == 3
        self.in_dim = in_dim
        self.views = 3  # views = 3, in_dim = [1000, 503, 1000]
        self.classes = num_class
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.att = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])  # fc [in_dim, in_dim]
        self.emb = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])  # fc [in_dim, hidden]
        self.aux_clf = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])  # fc [hidden, num_class]
        self.aux_conf = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])  # fc [hidden, 1]

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))  # [views*hidden, num_class]
        self.MMClasifier = nn.Sequential(*self.MMClasifier)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        # CLUE
        self.a2b = Prediction([hidden_dim[-1]]+prediction_dicts[0])
        self.a2c = Prediction([hidden_dim[-1]]+prediction_dicts[0])
        self.b2a = Prediction([hidden_dim[-1]]+prediction_dicts[1])
        self.b2c = Prediction([hidden_dim[-1]]+prediction_dicts[1])
        self.c2a = Prediction([hidden_dim[-1]]+prediction_dicts[2])
        self.c2b = Prediction([hidden_dim[-1]]+prediction_dicts[2])

        # head for instance-level CL, TODO

    def forward(self, data_list, label=None, infer=False, aux_loss=False, lambda_al=0.05):
        att_score, feat_emb, aux_logit, aux_confidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            att_score[view] = torch.sigmoid(self.att[view](data_list[view]))
            feat_emb[view] = data_list[view] * att_score[view]
            feat_emb[view] = F.dropout(F.relu(self.emb[view](feat_emb[view])), self.dropout, training=self.training)
            aux_logit[view] = self.aux_clf[view](feat_emb[view])
            aux_confidence[view] = self.aux_conf[view](feat_emb[view])
            feat_emb[view] = feat_emb[view] * aux_confidence[view]

        MMfeature = torch.cat([i for i in feat_emb.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit
        loss_dict = {}
        # 1. Loss between classifier and gt
        MMLoss = torch.mean(self.criterion(MMlogit, label))
        loss_dict["clf"] = round(MMLoss.item(), 4)

        if aux_loss:
            aux_losses = []
            for view in range(self.views):
                # 2. loss of attention scores, l0 norm
                MMLoss = MMLoss+torch.mean(att_score[view]) # L0 for attention scores
                pred = F.softmax(aux_logit[view], dim=1)
                p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
                # 3. confidence loss between calculated confidence and max classes in the aux classifiers
                # 4. loss for aux classifier
                confidence_loss = torch.mean(F.mse_loss(aux_confidence[view].view(-1), p_target)+self.criterion(aux_logit[view], label))
                MMLoss = MMLoss+ lambda_al * confidence_loss
                aux_losses.append(round(lambda_al * confidence_loss.item(), 4))

            loss_dict["aux"] = aux_losses
        return MMLoss, MMlogit, loss_dict


    def train_missing_cg(self, data_list, mask, label=None, device=None, 
                         aux_loss=False, lambda_al=0.05,
                         cross_omics_loss=False, lambda_col=0.05,
                         constrastive_loss=False, lambda_cl=0.05):
        # select the complete samples
        x1_train, x2_train, x3_train = data_list
        flag = torch.LongTensor([1, 1, 1]).to(device)
        flag = (mask == flag).int()
        flag = ((flag[:, 1] + flag[:, 0] + flag[:, 2]) == 3)
        train_view1, train_view2, train_view3 = x1_train[flag], x2_train[flag], x3_train[flag]
        data_list = [train_view1, train_view2, train_view3]
        label = label[flag]

        # perform feature embedding
        att_score, feat_emb, aux_logit, aux_confidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            att_score[view] = torch.sigmoid(self.att[view](data_list[view]))
            feat_emb[view] = data_list[view] * att_score[view]
            feat_emb[view] = F.dropout(F.relu(self.emb[view](feat_emb[view])),self.dropout, training=self.training)
            aux_logit[view] = self.aux_clf[view](feat_emb[view])
            aux_confidence[view] = self.aux_conf[view](feat_emb[view])
            feat_emb[view] = feat_emb[view] * aux_confidence[view]

        MMfeature = torch.cat([i for i in feat_emb.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)

        loss_dict = {}
        # 1. Loss between classifier and gt
        MMLoss = torch.mean(self.criterion(MMlogit, label))
        loss_dict["clf"] = round(MMLoss.item(), 4)
        if aux_loss:
            aux_losses = []
            for view in range(self.views):
                # 2. loss of attention scores, l0 norm
                MMLoss = MMLoss+torch.mean(att_score[view]) # L0 for attention scores
                pred = F.softmax(aux_logit[view], dim=1)
                p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
                # 3. confidence loss between calculated confidence and max classes in the aux classifiers
                # 4. loss for aux classifier
                confidence_loss = torch.mean(F.mse_loss(aux_confidence[view].view(-1), p_target)+self.criterion(aux_logit[view], label))
                MMLoss = MMLoss+ lambda_al * confidence_loss
                aux_losses.append(round(lambda_al * confidence_loss.item(), 4))
            loss_dict['aux_clf'] = aux_losses

        if cross_omics_loss:
             # cross-omics unified embedding
            a2b, _ = self.a2b(feat_emb[0])
            a2c, _ = self.a2c(feat_emb[0])
            b2a, _ = self.b2a(feat_emb[1])
            b2c, _ = self.b2c(feat_emb[1])
            c2a, _ = self.c2a(feat_emb[2])
            c2b, _ = self.c2b(feat_emb[2])
            pre1 = F.mse_loss(a2b, feat_emb[1])
            pre2 = F.mse_loss(b2a, feat_emb[0])
            pre3 = F.mse_loss(a2c, feat_emb[2])
            pre4 = F.mse_loss(c2a, feat_emb[0])
            pre5 = F.mse_loss(b2c, feat_emb[2])
            pre6 = F.mse_loss(c2b, feat_emb[1])

            loss_co = lambda_col * (pre1 + pre2 + pre3 + pre4 + pre5 + pre6)
            MMLoss = MMLoss +  loss_co
            loss_dict['col'] = round(loss_co.item(), 4)

        if constrastive_loss:
            loss_cil_ab = contrastive_Loss(feat_emb[0], feat_emb[1], lambda_cl)
            loss_cil_ac = contrastive_Loss(feat_emb[0], feat_emb[2], lambda_cl)
            loss_cil_bc = contrastive_Loss(feat_emb[1], feat_emb[2], lambda_cl)
            # loss_cil_ab = InfoNceDist()
            loss_cil = lambda_cl * (loss_cil_ab + loss_cil_ac + loss_cil_bc)
            MMLoss = MMLoss + loss_cil
            loss_dict['cil'] = round(loss_cil.item(), 4)

        return MMLoss, MMlogit, loss_dict

    def encode(self, data_x, f_att, f_emb, f_aux_conf):
        att_score = torch.sigmoid(f_att(data_x))
        feat_emb = data_x * att_score
        feat_emb = F.dropout(F.relu(f_emb(feat_emb)), self.dropout, training=False)
        aux_confidence = f_aux_conf(feat_emb)
        feat_emb = feat_emb*aux_confidence
        return feat_emb

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


    def infer_on_missing(self, data_list, mask, device):
        # make sure eval all modules before
        x1_train, x2_train, x3_train = data_list
        a_idx_eval = mask[:, 0] == 1
        b_idx_eval = mask[:, 1] == 1
        c_idx_eval = mask[:, 2] == 1
        a_missing_idx_eval = mask[:, 0] == 0
        b_missing_idx_eval = mask[:, 1] == 0
        c_missing_idx_eval = mask[:, 2] == 0

        # latent_code_x_eval, store information
        latent_code_a_eval = torch.zeros(x1_train.shape[0], self.hidden_dim[-1]).to(device)
        latent_code_b_eval = torch.zeros(x2_train.shape[0], self.hidden_dim[-1]).to(device)
        latent_code_c_eval = torch.zeros(x3_train.shape[0], self.hidden_dim[-1]).to(device)

        # predict on each omics without missing
        a_latent_eval = self.encode(x1_train[a_idx_eval], self.att[0], self.emb[0], self.aux_conf[0])
        b_latent_eval = self.encode(x2_train[b_idx_eval], self.att[1], self.emb[1], self.aux_conf[1])
        c_latent_eval = self.encode(x3_train[c_idx_eval], self.att[2], self.emb[2], self.aux_conf[2])

        if a_missing_idx_eval.sum() != 0:
            ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
            ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
            ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

            ano_bonlyhas = self.encode(x2_train[ano_bonlyhas_idx], self.att[1], self.emb[1], self.aux_conf[1])
            ano_bonlyhas, _ = self.b2a(ano_bonlyhas)
                           
            ano_conlyhas = self.encode(x3_train[ano_conlyhas_idx], self.att[2], self.emb[2], self.aux_conf[2])
            ano_conlyhas, _ = self.c2a(ano_conlyhas)

            ano_bcbothhas_1 = self.encode(x2_train[ano_bcbothhas_idx], self.att[1], self.emb[1], self.aux_conf[1])
            ano_bcbothhas_2 = self.encode(x3_train[ano_bcbothhas_idx], self.att[2], self.emb[2], self.aux_conf[2])
            ano_bcbothhas = (self.b2a(ano_bcbothhas_1)[0] + self.c2a(ano_bcbothhas_2)[0]) / 2.0

            latent_code_a_eval[ano_bonlyhas_idx] = ano_bonlyhas
            latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
            latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas
        
        if b_missing_idx_eval.sum() != 0:
            bno_aonlyhas_idx = b_missing_idx_eval * a_idx_eval * ~c_idx_eval
            bno_conlyhas_idx = b_missing_idx_eval * c_idx_eval * ~a_idx_eval
            bno_acbothhas_idx = b_missing_idx_eval * a_idx_eval * c_idx_eval

            bno_aonlyhas = self.encode(x1_train[bno_aonlyhas_idx], self.att[0], self.emb[0], self.aux_conf[0])
            bno_aonlyhas, _ = self.a2b(bno_aonlyhas)

            bno_conlyhas = self.encode(x3_train[bno_conlyhas_idx], self.att[2], self.emb[2], self.aux_conf[2])
            bno_conlyhas, _ = self.c2b(bno_conlyhas)

            bno_acbothhas_1 = self.encode(x1_train[bno_acbothhas_idx], self.att[0], self.emb[0], self.aux_conf[0])
            bno_acbothhas_2 = self.encode(x3_train[bno_acbothhas_idx], self.att[2], self.emb[2], self.aux_conf[2])
            bno_acbothhas = (self.a2b(bno_acbothhas_1)[0] + self.c2b(bno_acbothhas_2)[0]) / 2.0

            latent_code_b_eval[bno_aonlyhas_idx] = bno_aonlyhas
            latent_code_b_eval[bno_conlyhas_idx] = bno_conlyhas
            latent_code_b_eval[bno_acbothhas_idx] = bno_acbothhas

        if c_missing_idx_eval.sum() != 0:
            cno_aonlyhas_idx = c_missing_idx_eval * a_idx_eval * ~b_idx_eval
            cno_bonlyhas_idx = c_missing_idx_eval * b_idx_eval * ~a_idx_eval
            cno_abbothhas_idx = c_missing_idx_eval * a_idx_eval * b_idx_eval

            cno_aonlyhas = self.encode(x1_train[cno_aonlyhas_idx], self.att[0], self.emb[0], self.aux_conf[0])
            cno_aonlyhas, _ = self.a2c(cno_aonlyhas)

            cno_bonlyhas = self.encode(x2_train[cno_bonlyhas_idx], self.att[1], self.emb[1], self.aux_conf[1])
            cno_bonlyhas, _ = self.b2c(cno_bonlyhas)

            cno_abbothhas_1 = self.encode(x1_train[cno_abbothhas_idx], self.att[0], self.emb[0], self.aux_conf[0])
            cno_abbothhas_2 = self.encode(x2_train[cno_abbothhas_idx], self.att[1], self.emb[1], self.aux_conf[1])
            cno_abbothhas = (self.a2c(cno_abbothhas_1)[0] + self.b2c(cno_abbothhas_2)[0]) / 2.0

            latent_code_c_eval[cno_aonlyhas_idx] = cno_aonlyhas
            latent_code_c_eval[cno_bonlyhas_idx] = cno_bonlyhas
            latent_code_c_eval[cno_abbothhas_idx] = cno_abbothhas

        latent_code_a_eval[a_idx_eval] = a_latent_eval
        latent_code_b_eval[b_idx_eval] = b_latent_eval
        latent_code_c_eval[c_idx_eval] = c_latent_eval

        latent_fusion_train = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval], dim=1)
        MMlogit = self.MMClasifier(latent_fusion_train)
        return MMlogit