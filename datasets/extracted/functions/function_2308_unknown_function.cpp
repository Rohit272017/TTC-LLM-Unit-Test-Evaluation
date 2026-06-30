#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
namespace tensorflow {
REGISTER_OP("KmeansPlusPlusInitialization")
    .Input("points: float32")
    .Input("num_to_sample: int64")
    .Input("seed: int64")
    .Input("num_retries_per_sample: int64")
    .Output("samples: float32")
    .SetShapeFn(shape_inference::UnknownShape);
REGISTER_OP("KMC2ChainInitialization")
    .Input("distances: float32")
    .Input("seed: int64")
    .Output("index: int64")
    .SetShapeFn(shape_inference::ScalarShape);
REGISTER_OP("NearestNeighbors")
    .Input("points: float32")
    .Input("centers: float32")
    .Input("k: int64")
    .Output("nearest_center_indices: int64")
    .Output("nearest_center_distances: float32")
    .SetShapeFn(shape_inference::UnknownShape);
}  