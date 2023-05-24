#include "common.h"

void culling_gaussian_bsphere(Tensor mean, Tensor qvec, Tensor svec,
                              Tensor normal, Tensor pts, Tensor mask,
                              float thresh);