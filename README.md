==========================================
INTRODUCTION
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
DEFINING THE LEARNED ACTIVATION FUNCTIONS
==========================================
```
layers {<br />
  name: "learned1"<br />
  type: LEARNED_NEURON<br />
  bottom: "conv1"<br />
  top: "conv1"<br />
  learned_neuron_param {<br />
	sums: 5 # the value of S<br />
	# Initialize the "a" parameters. Each "a" is drawn from a uniform distribution between -0.2 and 0.2.<br />
	# the std increases as S decreases<br />
    	weight_filler1 {<br />
		type: "dense_uniform"<br />
     		std: 0.2<br />
    	}<br />
    	# Initialize the offset parameters "b." Each "b" is drawn from a gaussian distribution with standard deviation 0.5<br />
    	weight_filler2 {<br />
      		type: "gaussian"<br />
      		std: 0.5<br />
      		
    	}<br />
  }<br />
}
```
