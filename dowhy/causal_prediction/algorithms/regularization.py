import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dowhy.causal_prediction.algorithms.utils import mmd_compute

class Regularization:
    """
    Implements methods for applying unconditional and conditional regularization. 
    """

    def __init__(
        self, 
        E_conditioned, 
        ci_test,
        kernel_type,
        gamma,
    ):
        
        """
        :param E_conditioned: Binary flag indicating whether E-conditioned regularization has to be applied
        :param ci_test: Conditional independence metric used for regularization penalty. Currently, MMD is supported.
        :param kernel_type: Kernel type for MMD penalty. Currently, supports "gaussian" (RBF). If None, distance between mean and second-order statistics (covariances) is used.
        :param gamma: kernel bandwidth for MMD (due to implementation, the kernel bandwdith will actually be the reciprocal of gamma i.e., gamma=1e-6 implies kernel bandwidth=1e6. See `mmd_compute` in utils.py)
        """
        
        self.E_conditioned = E_conditioned  # E-conditioned regularization by default
        self.ci_test = ci_test
        self.kernel_type = kernel_type
        self.gamma = gamma
    
    def mmd(self, x, y):
        """
        Compute MMD penalty between x and y.

        """
        return mmd_compute(x, y, self.kernel_type, self.gamma)
     
    def unconditional_reg(self, classifs, attribute_labels, num_envs, E_eq_A=False): 
        """
        Implement unconditional regularization φ(x) ⊥⊥ A_i
        
        :param classifs: feature representations output from classifier layer (gφ(x))
        :param attribute_labels: attribute labels loaded with the dataset for attribute A_i
        :param num_envs: number of environments/domains
        :param E_eq_A: Binary flag indicating whether attribute (A_i) coinicides with environment (E) definition

        """
        
        penalty = 0
        
        if E_eq_A:  # Environment (E) and attribute (A) coincide
            if self.E_conditioned is False:  # there is no correlation between E and X_c 
                for i in range(num_envs):
                    for j in range(i + 1, num_envs):
                        penalty += self.mmd(classifs[i], classifs[j])

        else:
            if self.E_conditioned:
                for i in range(num_envs):
                    unique_attr_labels = torch.unique(attribute_labels[i])
                    unique_attr_label_indices = []
                    for label in unique_attr_labels:
                        label_ind = [ind for ind, j in enumerate(attribute_labels[i]) if j == label]
                        unique_attr_label_indices.append(label_ind)

                    nulabels = unique_attr_labels.shape[0]
                    for aidx in range(nulabels):
                        for bidx in range(aidx + 1, nulabels):
                            penalty += self.mmd(
                                classifs[i][unique_attr_label_indices[aidx]],
                                classifs[i][unique_attr_label_indices[bidx]],
                            )

            else:  # this currently assumes we have a disjoint set of attributes (Aind) across environments i.e., environment is defined by multiple closely related values of the attribute
                overall_nmb_indices, nmb_id = [], []
                for i in range(num_envs):
                    unique_attrs = torch.unique(attribute_labels[i])
                    unique_attr_indices = []
                    for attr in unique_attrs:
                        attr_ind = [ind for ind, j in enumerate(attribute_labels[i]) if j == attr]
                        unique_attr_indices.append(attr_ind)
                        overall_nmb_indices.append(attr_ind)
                        nmb_id.append(i)

                nuattr = len(overall_nmb_indices)
                for aidx in range(nuattr):
                    for bidx in range(aidx + 1, nuattr):
                        a_nmb_id = nmb_id[aidx]
                        b_nmb_id = nmb_id[bidx]
                        penalty += self.mmd(
                            classifs[a_nmb_id][overall_nmb_indices[aidx]],
                            classifs[b_nmb_id][overall_nmb_indices[bidx]],
                        )
                        
        return penalty

    def conditional_reg(self, classifs, attribute_labels, conditioning_subset, num_envs, E_eq_A=False): 
        """
        Implement conditional regularization φ(x) ⊥⊥ A_i | A_s
        
        :param classifs: feature representations output from classifier layer (gφ(x))
        :param attribute_labels: attribute labels loaded with the dataset for attribute A_i
        :param conditioning_subset: list of subset of observed variables A_s (attributes + targets) such that (X_c, A_i) are d-separated conditioned on this subset
        :param num_envs: number of environments/domains
        :param E_eq_A: Binary flag indicating whether attribute (A_i) coinicides with environment (E) definition

        Find group indices for conditional regularization based on conditioning subset by taking all possible combinations
        e.g., conditioning_subset = [A1, Y], where A1 is in {0, 1} and Y is in {0, 1, 2}, 
        we assign groups in the following way:
            A1 = 0, Y = 0 -> group 0
            A1 = 1, Y = 0 -> group 1
            A1 = 0, Y = 1 -> group 2
            A1 = 1, Y = 1 -> group 3
            A1 = 0, Y = 2 -> group 4
            A1 = 1, Y = 2 -> group 5
            
        Code snippet for computing group indices adapted from WILDS: https://github.com/p-lambda/wilds
            @inproceedings{wilds2021,
             title = {{WILDS}: A Benchmark of in-the-Wild Distribution Shifts},
             author = {Pang Wei Koh and Shiori Sagawa and Henrik Marklund and Sang Michael Xie and Marvin Zhang and Akshay Balsubramani and Weihua Hu and Michihiro Yasunaga and Richard Lanas Phillips and Irena Gao and Tony Lee and Etienne David and Ian Stavness and Wei Guo and Berton A. Earnshaw and Imran S. Haque and Sara Beery and Jure Leskovec and Anshul Kundaje and Emma Pierson and Sergey Levine and Chelsea Finn and Percy Liang},
             booktitle = {International Conference on Machine Learning (ICML)},
             year = {2021}
            }`
        
        """
        
        print("condirioning", len(conditioning_subset), len(conditioning_subset[0]), conditioning_subset[0][0].shape)
        
        grouping_data = torch.cat(conditioning_subset, 1) # list?
        assert grouping_data.min() >= 0, "Group numbers cannot be negative."
        cardinality = 1 + torch.max(grouping_data, dim=0)[0]
        cumprod = torch.cumprod(cardinality, dim=0)
        n_groups = cumprod[-1].item()
        factors_np = np.concatenate(([1], cumprod[:-1]))
        factors = torch.from_numpy(factors_np)
        group_indices = grouping_data @ factors

        penalty = 0
        
        if E_eq_A:  # Environment (E) and attribute (A) coincide
            if self.E_conditioned is False:  # there is no correlation between E and X_c 
                # TODO
                for i in range(num_envs):
                    for j in range(i + 1, num_envs):
                        penalty += self.mmd(classifs[i], classifs[j])

        else:
            if self.E_conditioned:
                for i in range(num_envs):
                    conditioning_subset_i = [subset_var[i] for subset_var in conditioning_subset]
                
        
        # if self.E_conditioned:
            
        # else:
            
        # if "causal" in self.attr_types:
        #     if self.E_conditioned:
                for i in range(nmb):
                    unique_labels = torch.unique(targets[i])  # find distinct labels in environment
                    unique_label_indices = []
                    for label in unique_labels:
                        label_ind = [ind for ind, j in enumerate(targets[i]) if j == label]
                        unique_label_indices.append(label_ind)

                    nulabels = unique_labels.shape[0]
                    for idx in range(nulabels):
                        unique_attrs = torch.unique(
                            causal_attribute_labels[i][unique_label_indices[idx]]
                        )                             # find distinct attributes in environment with same label
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
                                penalty_causal += self.mmd(
                                    classifs[i][unique_attr_indices[aidx]], classifs[i][unique_attr_indices[bidx]]
                                )

            else:
                overall_label_attr_vindices = {}  # storing attribute indices
                overall_label_attr_eindices = {}  # storing corresponding environment indices

                for i in range(num_envs):
                    unique_labels = torch.unique(targets[i])  # find distinct labels in environment
                    unique_label_indices = []
                    for label in unique_labels:
                        label_ind = [ind for ind, j in enumerate(targets[i]) if j == label]
                        unique_label_indices.append(label_ind)

                    nulabels = unique_labels.shape[0]
                    for idx in range(nulabels):
                        label = unique_labels[idx]
                        if label not in overall_label_attr_vindices:
                            overall_label_attr_vindices[label] = {}
                            overall_label_attr_eindices[label] = {}

                        unique_attrs = torch.unique(
                            causal_attribute_labels[i][unique_label_indices[idx]]
                        )  # find distinct attributes in environment with same label
                        unique_attr_indices = []
                        for attr in unique_attrs:  # storing indices with same attribute value and label
                            if attr not in overall_label_attr_vindices[label]:
                                overall_label_attr_vindices[label][attr] = []
                                overall_label_attr_eindices[label][attr] = []
                            single_attr = []
                            for y_attr_idx in unique_label_indices[idx]:
                                if causal_attribute_labels[i][y_attr_idx] == attr:
                                    single_attr.append(y_attr_idx)
                            overall_label_attr_vindices[label][attr].append(single_attr)
                            overall_label_attr_eindices[label][attr].append(i)
                            unique_attr_indices.append(single_attr)

                for (
                    y_val
                ) in (
                    overall_label_attr_vindices
                ):  # applying MMD penalty between distributions P(φ(x)|ai, y), P(φ(x)|aj, y) i.e samples with different attribute values but same label
                    tensors_list = []
                    for attr in overall_label_attr_vindices[y_val]:
                        attrs_list = []
                        if overall_label_attr_vindices[y_val][attr] != []:
                            for il_ind, indices_list in enumerate(overall_label_attr_vindices[y_val][attr]):
                                attrs_list.append(
                                    classifs[overall_label_attr_eindices[y_val][attr][il_ind]][indices_list]
                                )
                        if len(attrs_list) > 0:
                            tensor_attrs = torch.cat(attrs_list, 0)
                            tensors_list.append(tensor_attrs)

                    nuattr = len(tensors_list)
                    for aidx in range(nuattr):
                        for bidx in range(aidx + 1, nuattr):
                            penalty_causal += self.mmd(tensors_list[aidx], tensors_list[bidx])

        # Aind regularization
        if "ind" in self.attr_types:
            if self.E_eq_Aind:  # Environment (E) and Independent attribute (Aind) coincide
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
                                penalty_ind += self.mmd(
                                    classifs[i][unique_aind_label_indices[aidx]],
                                    classifs[i][unique_aind_label_indices[bidx]],
                                )

                else:  # this currently assumes we have a disjoint set of attributes (Aind) across environments i.e., environment is defined by multiple closely related values of the attribute
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
                            penalty_ind += self.mmd(
                                classifs[a_nmb_id][overall_nmb_indices[aidx]],
                                classifs[b_nmb_id][overall_nmb_indices[bidx]],
                            )
