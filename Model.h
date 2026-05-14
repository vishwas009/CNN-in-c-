#pragma once
#include <deque>
#include <chrono>
#include "layer.h"

float Binary_Cross_Entropy_Cost(const float* hx, const float* _Y, const int n);
float Mean_Squared_Error(const float* hx, const float* _Y, const int n);
float Mean_Absolute_Error(const float* hx, const float* _Y, const int n);
float Categorical_Cross_Entropy_Cost(const float* hx, const float* _Y, const int n);


struct Univ_Layer_Param {
	LAYER_TYPE l_type;
	int params[5] = { 0 };
};

class Model
{
private:
	bool m_init;
	OPTIMIZER m_Opt;
	std::deque<Layer_Base*> model_layers;
	std::deque<Univ_Layer_Param> model_layers_param;
	Eigen::VectorXi m_in_shape;
	float (*cost_Function)(const float*, const float*, const int);

public:
	Model(OPTIMIZER opt_algo, LOSS_FUNCTION loss, Eigen::VectorXi input_shape) :m_init(false), m_Opt(opt_algo)
	{
		m_in_shape = input_shape;
		switch (loss) {
		case LOSS_FUNCTION::MSE: {cost_Function = Mean_Squared_Error; break; }
		case LOSS_FUNCTION::MAE: {cost_Function = Mean_Absolute_Error; break; }
		case LOSS_FUNCTION::BINARY_CROSS_ENTROPY:{cost_Function = Binary_Cross_Entropy_Cost; break; }
		case LOSS_FUNCTION::CATEGORICAL_CROSS_ENTROPY:{cost_Function = Categorical_Cross_Entropy_Cost; break; }
		default: {cost_Function = nullptr; break; }
		}
	}

	virtual ~Model() {
		/// Todo Proper Destruction ///
		int n = model_layers.size();
		for (int i = 0; i < n; i++) {
			delete model_layers[i];
		}
	}

	void add_Conv2D_Layer(int fx, int fy, int num_filter, ACT_TYPE act_f, REGULARIZATION reg) {
		Univ_Layer_Param temp;
		temp.l_type = LAYER_TYPE::CONV;
		temp.params[0] = fx; temp.params[1] = fy; temp.params[2] = num_filter;
		temp.params[3] = static_cast<int>(act_f);
		temp.params[4] = static_cast<int>(reg);

		model_layers_param.push_back(temp);
	}

	void add_Max_Pool_2D_layer(int fx = 2, int fy = 2, int stride = 1) {
		Univ_Layer_Param temp;
		temp.l_type = LAYER_TYPE::MAX_POOL;
		temp.params[0] = fx; temp.params[1] = fy; temp.params[2] = stride; 

		model_layers_param.push_back(temp);
	}

	void add_Dense_Layer(int out_size, ACT_TYPE act_f, REGULARIZATION reg) {
		Univ_Layer_Param temp;
		temp.l_type = LAYER_TYPE::DENSE;
		temp.params[0] = out_size;
		temp.params[1] = static_cast<int>(act_f);
		temp.params[2] = static_cast<int>(reg);

		model_layers_param.push_back(temp);
	}

	bool Initialize();
	float model_Forward_Pass(Tensor4f& _X, const Tensor4f& _Y);  // Overloads maybe ///
	bool Run(Tensor4f& _X, const Tensor4f& _Y, int batch_size, int epochs);
	void Model_Summary();

};

