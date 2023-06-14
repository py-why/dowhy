"""Beam search utilities for optimization.

This module implements the boolean ruleset estimator from OverRule [1]. Code is adapted (with some simplifications)
from https://github.com/clinicalml/overlap-code, under the MIT License.

[1] Oberst, M., Johansson, F., Wei, D., Gao, T., Brat, G., Sontag, D., & Varshney, K. (2020). Characterization of
Overlap in Observational Studies. In S. Chiappa & R. Calandra (Eds.), Proceedings of the Twenty Third International
Conference on Artificial Intelligence and Statistics (Vol. 108, pp. 788–798). PMLR. https://arxiv.org/abs/1907.04138
"""

import numpy as np
import pandas as pd


class PricingInstance:
    """
    Instance of the pricing problem.

    For more details, see:

    Dash, S., Gunluk, O., and Wei, D. (2018). Boolean decision rules via column generation.
    In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa- Bianchi, N.,
    and Garnett, R., editors, Advances in Neural Information Processing Systems 31,
    pages 4660–4670. Curran Associates, Inc.
    """

    def __init__(self, rp, rn, Xp, Xn, v0, z0):
        self.rp = rp
        self.rn = rn
        self.Xp = Xp
        self.Xn = Xn
        self.v0 = v0
        self.z0 = z0

    def eval_singletons(self, lambda1):
        """Evaluate all singleton solutions (adding one literal)"""
        self.Rp = np.dot(self.rp, self.Xp)
        self.Rn = np.dot(self.rn, self.Xn)
        self.v1 = self.v0 - self.Rp - self.Rn + lambda1
        self.v1 = pd.Series(self.v1, index=self.Xp.columns)

    def compute_LB(self, lambda1):
        """Compute lower bound on higher-order solutions"""
        Rp0 = self.rp.sum()
        self.LB = np.minimum(np.cumsum(np.sort(self.Rp)[::-1])[1:], Rp0)
        self.LB += np.sort(self.Rn)[-2::-1]
        self.LB -= lambda1 * np.arange(2, len(self.Rp) + 1)
        self.LB = self.v0 - self.LB

        # Lower bound specific to each singleton solution
        self.LB1 = self.v1 + self.Rp - Rp0 + lambda1
        if len(self.LB):
            self.LB1[self.LB1 < self.LB.min()] = self.LB.min()


def beam_search(
    r,
    X,
    lambda0: float,
    lambda1: float,
    K: int = 1,
    UB: float = 0,
    D: int = 10,
    B: int = 5,
    wLB: float = 0.5,
    eps: float = 1e-6,
):
    """
    Beam search to generate solutions to pricing problem.

    :param r: Cost vector (residuals)
    :param X: Binary features in a DataFrame
    :param lambda0: Fixed cost of a term
    :type lambda0: float
    :param lambda1: Cost per literal
    :type lambda1: float
    :param K: Maximum number of solutions returned, defaults to 1
    :type K: int, optional
    :param UB: Initial upper bound on value of solutions, defaults to 0
    :type UB: float, optional
    :param D: Maximum Degree, defaults to 10
    :type D: int, optional
    :param B: Beam width, defaults to 5
    :type B: int, optional
    :param wLB: Weight on lower bound in evaluating nodes, defaults to 0.5
    :type wLB: float, optional
    :param eps: Numerical tolerance on comparisons, defaults to 1e-6
    :type eps: float, optional
    """

    # Initialize output
    vOut = np.array([])
    zOut = pd.DataFrame(index=X.columns)

    # Remove redundant rows by grouping by unique feature combinations and summing residual
    X2 = X.copy()
    r2 = r

    # Initialize queue with root instance
    # Separate data according to positive and negative residuals
    rp = r2[r2 > 0]
    rn = r2[r2 < 0]
    Xp = 1 - X2.loc[r2 > 0]
    Xn = 1 - X2.loc[r2 < 0]
    instCurr = [PricingInstance(rp, rn, Xp, Xn, r2.sum() + lambda0, pd.Series(0, index=zOut.index))]

    # Iterate over increasing degree while queue is non-empty
    deg = 0
    while len(instCurr) and deg < D:
        deg += 1

        # Initialize list of children to process
        vNext = np.array([])
        vNextMax = np.inf
        zNext = pd.DataFrame([], index=X2.columns)
        instNext = []

        # Process instances in queue
        for inst in instCurr:
            # inst = instCurr[0]

            # Evaluate all singleton solutions
            inst.eval_singletons(lambda1)

            # Best solutions that also improve on current output (allow for duplicate removal)
            vCand = inst.v1[inst.v1 < UB - eps].sort_values()[: K + B]
            if len(vCand):
                zCand = pd.DataFrame(zOut.index.values[:, np.newaxis] == vCand.index.values, index=zOut.index).astype(
                    int
                )
                zCand = zCand.add(inst.z0, axis=0)
                # Append to current output
                vOut = np.append(vOut, vCand.values)
                zOut = pd.concat([zOut, zCand], axis=1, ignore_index=True)
                # Remove duplicates
                _, idxUniq = np.unique(zOut, return_index=True, axis=1)
                vOut = vOut[idxUniq]
                zOut = zOut.iloc[:, idxUniq]
                # Update output
                indBest = np.argsort(vOut)[:K]
                vOut = vOut[indBest]
                UB = vOut.max()
                zOut = zOut.iloc[:, indBest]
                zOut.columns = range(zOut.shape[1])

            # Compute lower bounds on higher-degree solutions
            inst.compute_LB(lambda1)

            # Evaluate children using weighted average of their costs and LBs
            vChild = (1 - wLB) * inst.v1 + wLB * inst.LB1
            # Best children with potential to improve on current output and current candidates (allow for duplicate removal)
            vChild = vChild[(inst.LB1 < UB - eps) & (vChild < vNextMax - eps)].sort_values()[: 2 * B]
            # Iterate through best children
            numAdded = 0
            for i in vChild.index:
                # New "zero" solution
                z0 = inst.z0.copy()
                z0[i] = 1
                # Check if duplicate
                if zNext.eq(z0, axis=0).all().any():
                    continue
                # Add to candidates for further processing
                vNext = np.append(vNext, vChild[i])
                zNext = pd.concat([zNext, z0], axis=1, ignore_index=True)
                # Create pricing instance
                # Remove covered rows
                rowKeep = inst.Xp[i] == 0
                rp = inst.rp[rowKeep]
                Xp = inst.Xp.loc[rowKeep]
                rowKeep = inst.Xn[i] == 0
                rn = inst.rn[rowKeep]
                Xn = inst.Xn.loc[rowKeep]
                # Remove redundant features
                colKeep = pd.Series(Xp.columns.get_level_values(0) != i[0], index=Xp.columns)
                if i[1] == "<=":
                    thresh = Xp[i[0]].columns.get_level_values(1).to_series().replace("NaN", np.nan)
                    colKeep[i[0]] = (Xp[i[0]].columns.get_level_values(0) == ">") & (thresh < i[2])
                elif i[1] == ">":
                    thresh = Xp[i[0]].columns.get_level_values(1).to_series().replace("NaN", np.nan)
                    colKeep[i[0]] = (Xp[i[0]].columns.get_level_values(0) == "<=") & (thresh > i[2])
                elif i[1] == "!=":
                    colKeep[i[0]] = (Xp[i[0]].columns.get_level_values(0) == "!=") & (
                        Xp[i[0]].columns.get_level_values(1) != i[2]
                    )
                Xp = Xp.loc[:, colKeep]
                Xn = Xn.loc[:, colKeep]
                instNext.append(PricingInstance(rp, rn, Xp, Xn, inst.v1[i], z0))
                # Track number of candidates added
                numAdded += 1
                if numAdded == B:
                    break

            # Update candidates
            indBest = np.argsort(vNext)[:B]
            vNext = vNext[indBest]
            if len(vNext):
                vNextMax = vNext[-1]
            instNext = [instNext[i] for i in indBest]

        # Instances to process in next iteration
        instCurr = instNext

    # Conjunctions corresponding to solutions
    aOut = 1 - (np.dot(1 - X, zOut) > 0)

    return vOut, zOut, aOut
