import torch
from torch.nn import functional as F
from torch import nn
from dowhy.causal_prediction.algorithms.base_algorithm import Algorithm
from dowhy.causal_prediction.algorithms.utils import mmd_compute

class CACM(Algorithm): 
    def __init__(self, model, optimizer='Adam', lr=1e-3, weight_decay=0., betas=(0.9, 0.999), momentum=0.9,
            kernel_type='gaussian', ci_test='mmd', attr_types=[], E_conditioned=True, E_eq_Aind=True, 
            gamma=1e-6, lambda_causal=1.0, lambda_ind=1.0):
        super().__init__(model, optimizer, lr, weight_decay, betas, momentum)

        
        self.kernel_type = kernel_type
        self.attr_types = attr_types
        self.E_conditioned = E_conditioned # E-conditioned regularization by default
        self.E_eq_Aind = E_eq_Aind
        self.gamma = gamma
        self.lambda_causal = lambda_causal
        self.lambda_ind = lambda_ind

    def mmd(self, x, y):
        return mmd_compute(x, y, self.kernel_type, self.gamma)

    def training_step(self, train_batch, batch_idx):

        self.featurizer = self.model[0]
        self.classifier = self.model[1]
    
        minibatches = train_batch

        objective = 0
        correct, total = 0, 0
        penalty_causal, penalty_ind = 0, 0
        nmb = len(minibatches)

        if len(minibatches[0]) == 4:
            features = [self.featurizer(xi) for xi, _, _, _ in minibatches]
            classifs = [self.classifier(fi) for fi in features]
            targets = [yi for _, yi, _, _ in minibatches]
            causal_attribute_labels = [ai for _, _, ai, _ in minibatches]
            ind_attribute_labels = [ai for _, _, _, ai in minibatches]
        elif len(minibatches[0]) == 3: # redundant for now since enforcing 4-dim output from dataset
            features = [self.featurizer(xi) for xi, _, _ in minibatches]
            classifs = [self.classifier(fi) for fi in features]
            targets = [yi for _, yi, _ in minibatches]
            causal_attribute_labels = [ai for _, _, ai in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            correct += (torch.argmax(classifs[i], dim=1) == targets[i]).float().sum().item()
            total += classifs[i].shape[0]

        # Acause regularization
        if 'causal' in self.attr_types:
            if self.E_conditioned:
                for i in range(nmb):
                    unique_labels = torch.unique(targets[i])
                    unique_label_indices = []
                    for label in unique_labels:
                        label_ind = [ind for ind, j in enumerate(targets[i]) if j == label]
                        unique_label_indices.append(label_ind)

                    nulabels = unique_labels.shape[0]
                    for idx in range(nulabels):
                        unique_attrs = torch.unique(causal_attribute_labels[i][unique_label_indices[idx]])
                        unique_attr_indices = []
                        for attr in unique_attrs:
                            single_attr = []
                            for y_attr_idx in unique_label_indices[idx]:
                                if causal_attribute_labels[i][y_attr_idx] == attr:
                                    single_attr.append(y_attr_idx)
                            unique_attr_indices.append(single_attr)

                        nuattr = unique_attrs.shape[0]
                        for aidx in range(nuattr):
                            for bidx in range(aidx + 1, nuattr):
                                penalty_causal += self.mmd(classifs[i][unique_attr_indices[aidx]], classifs[i][unique_attr_indices[bidx]])

            else: #TODO
                pass

        # Aind regularization
        if 'ind' in self.attr_types: 
            if self.E_eq_Aind: # Environment (E) and Independent attribute (Aind) coincide
                for i in range(nmb):
                    for j in range(i + 1, nmb):
                        penalty_ind += self.mmd(classifs[i], classifs[j])

            else:
                if self.E_conditioned: 
                    for i in range(nmb):
                        unique_aind_labels = torch.unique(ind_attribute_labels[i])
                        unique_aind_label_indices = []
                        for label in unique_aind_labels:
                            label_ind = [ind for ind, j in enumerate(ind_attribute_labels[i]) if j == label]
                            unique_aind_label_indices.append(label_ind)

                        nulabels = unique_aind_labels.shape[0]
                        for aidx in range(nulabels):
                            for bidx in range(aidx + 1, nulabels):
                                penalty_ind += self.mmd(classifs[i][unique_aind_label_indices[aidx]], classifs[i][unique_aind_label_indices[bidx]])
            
                else:
                    overall_nmb_indices, nmb_id = [], []
                    for i in range(nmb):
                        unique_attrs = torch.unique(ind_attribute_labels[i])
                        unique_attr_indices = []
                        for attr in unique_attrs:
                            attr_ind = [ind for ind, j in enumerate(ind_attribute_labels[i]) if j == attr]
                            unique_attr_indices.append(attr_ind)
                            overall_nmb_indices.append(attr_ind)
                            nmb_id.append(i)

                    nuattr = len(overall_nmb_indices)
                    for aidx in range(nuattr):
                        for bidx in range(aidx + 1, nuattr):
                            a_nmb_id = nmb_id[aidx]
                            b_nmb_id = nmb_id[bidx]
                            penalty_ind += self.mmd(classifs[a_nmb_id][overall_nmb_indices[aidx]], classifs[b_nmb_id][overall_nmb_indices[bidx]])
                                
        objective /= nmb
        if nmb > 1:
            penalty_causal /= (nmb * (nmb - 1) / 2)
            penalty_ind /= (nmb * (nmb - 1) / 2)

        # Compile loss
        loss = objective
        loss += self.lambda_causal * penalty_causal
        loss += self.lambda_ind * penalty_ind

        if torch.is_tensor(penalty_causal):
            penalty_causal = penalty_causal.item()
            self.log('penalty_causal', penalty_causal, on_step=False, on_epoch=True, prog_bar=True)
        if torch.is_tensor(penalty_ind):
            penalty_ind = penalty_ind.item()
            self.log('penalty_ind', penalty_ind, on_step=False, on_epoch=True, prog_bar=True)

        acc = correct / total

        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
        
    



    

    