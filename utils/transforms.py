import numpy as np
import torch
import torch.nn.functional as F
import kornia
from kornia.geometry.conversions import QuaternionCoeffOrder
from torchtyping import TensorType


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


@torch.no_grad()
def qsvec2rotmat_batched(
    qvec: TensorType["N", 4], svec: TensorType["N", 3]
) -> TensorType["N", 3, 3]:
    unscaled_rotmat = kornia.geometry.conversions.quaternion_to_rotation_matrix(
        qvec, QuaternionCoeffOrder.WXYZ
    )

    # TODO: check which I current think that scale should be copied row-wise since in eq (6) the S matrix is right-hand multplied to R
    rotmat = svec.unsqueeze(-2) * unscaled_rotmat

    return rotmat