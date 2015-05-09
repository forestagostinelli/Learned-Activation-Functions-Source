==========================================
Introduction
==========================================
This repository if for the reproduction of research by Agostinelli et al. Learning Activation Functions to Improve Deep Neural Networks. http://arxiv.org/abs/1412.6830

This is a learnable activation function for neural networks. Experiments have shown that it outperforms rectified linear units (ReLU) and Leaky ReLU.

==========================================
Memory
==========================================
The use of this layer will require more memory. There is a "save_mem" option that can be used. However, this will lead to slower performance and has not yet been thoroughly tested.

==========================================
In place computation
==========================================
In place computation can be done. However, due to implementation details, it does not conserve memory and tests show it will result in a slight decrease in speed.

==========================================
Solver Files
==========================================
We made custom changes to the solver files. The changes are reflected in src/caffe/solver.cpp, include/caffe.solver.hpp, and src/caffe/proto/caffe.proto

==========================================
Known Bugs
==========================================
When a dropout layer comes immediately after an APL layer do not use in-place blobs for either the APL layer or the dropout layer.

For example, if the blob coming into apl is "conv1"
then for APL layer bottom is "conv1" and top is "conv1_a"
For dropout bottom is "conv1_l" and top is "conv1_d"

This is only seen in the new version of Caffe and the cause is not yet known.

==========================================
Defining the Learned Activation Functions
==========================================
```
layer {
  name: "apl1"
  type: "APL"
  bottom: "blob_name"
  top: "blob_name"
  param {
    decay_mult: 1 # We set so weight decay is 0.001
  }
  param {
    decay_mult: 1 # We set so weight decay is 0.001
  }
  apl_param {
    sums: 1
    slope_filler {
      type: "uniform"
      min: -0.5
      max: 0.5
    }
    offset_filler {
      type: "gaussian"
      std: 0.5
    }
  }
}
```
