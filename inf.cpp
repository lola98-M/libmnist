#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

int main() {
	//Net model;	
	torch::jit::script::Module model;
	std::string module_path = "pytorch_mnist.pt";
	model=  torch::jit::load(module_path);
	
	// Create a vector of inputs.
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(torch::ones({1, 1, 28, 28}));

	// Execute the model and turn its output into a tensor.
	at::Tensor output = model.forward(inputs).toTensor();
	std::cout << output << std::endl;	
    
return 0;
}	
