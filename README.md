==========================================
INTRODUCTION
==========================================
This repository if for the reproduction of research by Agostinelli et al. Learning Activation Functions to Improve Deep Neural Networks. http://arxiv.org/abs/1412.6830

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
layers {

  name: "learned1"
  
  type: LEARNED_NEURON
  
  bottom: "conv1"
  
  top: "conv1"
  
  blobs_lr: 1 
  
  blobs_lr: 1
  
  learned_neuron_param {
  
	sums: 5 # the value of S
		
	# Initialize the "a" parameters. Each "a" is drawn from a uniform distribution between -0.2 and 0.2.
	# the std increases as S decreases
		
    	weight_filler1 {
    	
		type: "dense_uniform"
		
     		std: 0.2
     		
    	}
    	# Initialize the offset parameters "b." Each "b" is drawn from a gaussian distribution with standard deviation 0.5
    	weight_filler2 {
    	
      		type: "gaussian"
      		
      		std: 0.5
      		
    	}
  }
  
}
