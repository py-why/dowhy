import torch
from torch import tensor

from dowhy.causal_prediction.algorithms.utils import gaussian_kernel, mmd_compute


class Regularizer:
    """
    Implements methods for applying unconditional and conditional regularization.
    Optimized for GPU throughput by minimizing Python control flow.
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

    def _optimized_mmd_penalty(self, list_of_feature_tensors: list):
        """
        Computes the sum of MMD penalties between all pairs in a list of tensors.
        This optimized version handles both 'gaussian' and other kernel types efficiently,
        avoiding redundant computations while maintaining backward compatibility.

        :param list_of_feature_tensors: A list where each element is a feature tensor
                                        corresponding to a unique attribute value.
        :return: The total MMD penalty as a scalar tensor.
        """
        valid_tensors = [t.double() for t in list_of_feature_tensors if t.shape[0] > 0]
        k = len(valid_tensors)

        if k <= 1:
            return 0.0

        original_dtype = list_of_feature_tensors[0].dtype
        original_device = list_of_feature_tensors[0].device

        if self.kernel_type == "gaussian":
            sum_K_ii = sum(gaussian_kernel(t, t, self.gamma).mean() for t in valid_tensors)
            sum_K_ij = 0
            for i in range(k):
                for j in range(i + 1, k):
                    sum_K_ij += gaussian_kernel(valid_tensors[i], valid_tensors[j], self.gamma).mean()
            penalty = (k - 1) * sum_K_ii - 2 * sum_K_ij

        else:
            means = [t.mean(0, keepdim=True) for t in valid_tensors]
            cents = [t - m for t, m in zip(valid_tensors, means)]

            covas = []
            for t, cent in zip(valid_tensors, cents):
                n = t.shape[0]
                d_dim = t.shape[1]
                if n > 1:
                    covas.append((cent.t() @ cent) / (n - 1))
                else:
                    covas.append(torch.zeros_likes((d_dim, d_dim), device=original_device, dtype=torch.float64))

            penalty = tensor(0.0, device=original_device, dtype=torch.float64)
            for i in range(k):
                for j in range(i + 1, k):
                    mean_diff = (means[i] - means[j]).pow(2).mean()
                    cova_diff = (covas[i] - covas[j]).pow(2).mean()
                    penalty += mean_diff + cova_diff

        return penalty.to(dtype=original_dtype)

    def _split_by_attribute(self, features, labels):
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return [features]

        return [features[labels == label] for label in unique_labels]

    def unconditional_reg(self, classifs, attribute_labels, num_envs, E_eq_A=False, use_optimization=False):
        """
        Implement unconditional regularization φ(x) ⊥⊥ A_i

        :param classifs: feature representations output from classifier layer (gφ(x))
        :param attribute_labels: attribute labels loaded with the dataset for attribute A_i
        :param num_envs: number of environments/domains
        :param E_eq_A: Binary flag indicating whether attribute (A_i) coincides with environment (E) definition
        :param use_optimization: If True, uses an algebraically optimized method to compute the penalty, which is
        faster and suitable for standard MMD computations. If False, uses the original nested loop, which is more
        extensible for implementing new or custom MMD variants, but may be slower. Choose True for performance
        with standard MMD, and False when correctness or extensibility for new MMD types is required.


        """

        penalty = tensor(0.0, dtype=classifs[0].dtype, device=classifs[0].device)

        if E_eq_A:  # Environment (E) and attribute (A) coincide
            if self.E_conditioned is False:  # there is no correlation between E and X_c
                if use_optimization:
                    penalty += self._optimized_mmd_penalty(classifs)
                else:
                    for i in range(num_envs):
                        for j in range(i + 1, num_envs):
                            penalty += self.mmd(classifs[i], classifs[j])

        else:
            if self.E_conditioned:
                for i in range(num_envs):
                    tensors_list = self._split_by_attribute(classifs[i], attribute_labels[i])

                    if use_optimization:
                        penalty += self._optimized_mmd_penalty(tensors_list)
                    else:
                        k = len(tensors_list)
                        for aidx in range(k):
                            for bidx in range(aidx + 1, k):
                                penalty += self.mmd(tensors_list[aidx], tensors_list[bidx])

            else:
                all_features = torch.cat(classifs, dim=0)
                all_labels = torch.cat(attribute_labels, dim=0)

                tensors_list = self._split_by_attribute(all_features, all_labels)

                if use_optimization:
                    penalty += self._optimized_mmd_penalty(tensors_list)
                else:
                    k = len(tensors_list)
                    for aidx in range(k):
                        for bidx in range(aidx + 1, k):
                            penalty += self.mmd(tensors_list[aidx], tensors_list[bidx])

        return penalty

    def conditional_reg(
        self, classifs, attribute_labels, conditioning_subset, num_envs, E_eq_A=False, use_optimization=False
    ):
        """
        Implement conditional regularization φ(x) ⊥⊥ A_i | A_s

        :param classifs: feature representations output from classifier layer (gφ(x))
        :param attribute_labels: attribute labels loaded with the dataset for attribute A_i
        :param conditioning_subset: list of subset of observed variables A_s (attributes + targets) such that (X_c, A_i) are d-separated conditioned on this subset
        :param num_envs: number of environments/domains
        :param E_eq_A: Binary flag indicating whether attribute (A_i) coincides with environment (E) definition
        :param use_optimization: If True, uses an algebraically optimized method to compute the penalty.
                                 If False, uses the original nested loop, which is more extensible for new MMD

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
        penalty = tensor(0.0, dtype=classifs[0].dtype, device=classifs[0].device)

        if E_eq_A:  # Environment (E) and attribute (A) coincide
            if self.E_conditioned is False:
                all_feats = []
                all_groups = []
                all_attrs = []

                for i in range(num_envs):
                    cond_subset_i = [subset_var[i] for subset_var in conditioning_subset]
                    cond_subset_i_uniform = [ele.unsqueeze(1) if ele.dim() == 1 else ele for ele in cond_subset_i]
                    if cond_subset_i_uniform:
                        group_data_i = torch.cat(cond_subset_i_uniform, 1)
                    else:
                        group_data_i = torch.zeros((classifs[i].shape[0], 1), device=classifs[i].device)

                    all_feats.append(classifs[i])
                    all_groups.append(group_data_i)
                    all_attrs.append(torch.full((classifs[i].shape[0],), i, device=classifs[i].device))

                total_feats = torch.cat(all_feats, 0)
                total_groups = torch.cat(all_groups, 0)
                total_attrs = torch.cat(all_attrs, 0)

                penalty += self._compute_conditional_penalty(total_feats, total_attrs, total_groups, use_optimization)

        else:
            if self.E_conditioned:
                for i in range(num_envs):
                    cond_subset_i = [subset_var[i] for subset_var in conditioning_subset]
                    cond_subset_i_uniform = [ele.unsqueeze(1) if ele.dim() == 1 else ele for ele in cond_subset_i]
                    if cond_subset_i_uniform:
                        group_data = torch.cat(cond_subset_i_uniform, 1)
                    else:
                        group_data = torch.zeros((classifs[i].shape[0], 1), device=classifs[i].device)

                    penalty += self._compute_conditional_penalty(
                        classifs[i], attribute_labels[i], group_data, use_optimization
                    )

            else:
                all_feats = torch.cat(classifs, 0)
                all_attrs = torch.cat(attribute_labels, 0)

                all_cond_vars = []
                for var_list in conditioning_subset:
                    all_cond_vars.append(torch.cat(var_list, 0))

                cond_uniform = [ele.unsqueeze(1) if ele.dim() == 1 else ele for ele in all_cond_vars]
                if cond_uniform:
                    total_groups = torch.cat(cond_uniform, 1)
                else:
                    total_groups = torch.zeros((all_feats.shape[0], 1), device=all_feats.device)

                penalty += self._compute_conditional_penalty(all_feats, all_attrs, total_groups, use_optimization)

        return penalty

    def _compute_conditional_penalty(self, features, attributes, group_data, use_optimization):
        penalty = tensor(0.0, dtype=features.dtype, device=features.device)

        unique_groups, group_indices = torch.unique(group_data, dim=0, return_inverse=True)
        present_group_ids = torch.unique(group_indices)

        for gid in present_group_ids:
            mask = group_indices == gid
            group_feats = features[mask]
            group_attrs = attributes[mask]
            tensors_list = self._split_by_attribute(group_feats, group_attrs)

            if len(tensors_list) > 1:
                if use_optimization:
                    penalty += self._optimized_mmd_penalty(tensors_list)
                else:
                    k = len(tensors_list)
                    for aidx in range(k):
                        for bidx in range(aidx + 1, k):
                            penalty += self.mmd(tensors_list[aidx], tensors_list[bidx])

        return penalty
