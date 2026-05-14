#pragma once

#include <iostream>
#include "common.h"
#include "activations.h"


class Layer_Base {
public:
	Layer_Base(){}
	virtual ~Layer_Base(){}

	virtual bool Forward_Pass(Tensor4f& _X, Tensor4f& _Z) = 0;
	virtual bool Backward_Pass(Tensor4f& dout, Tensor4f& dX) = 0;
	virtual void Update_Weights(const float alpha, REGULARIZATION l_type) = 0;
	virtual LAYER_TYPE Type() const = 0;
	virtual Eigen::VectorXi Out_shape(const Eigen::VectorXi in_shape) = 0;
	virtual void Summary() = 0;
};

class Dense_FC : public Layer_Base {
private:
	int layer_in;
	int layer_out;
	Activations* act_func;
	REGULARIZATION reg_type;

	// Refrences for Backward Pass //
	Tensor4f* ref_X;
	Tensor4f* ref_Z;

	// Layer Weights and Gradients //
	Matrix<float, Dynamic, Dynamic, RowMajor> _W;
	Matrix<float, 1, Dynamic, RowMajor> _b;
	Matrix<float, Dynamic, Dynamic, RowMajor> grads_W;
	Matrix<float, 1, Dynamic, RowMajor> grads_b;

public:
	Dense_FC(int in_size, int out_size, ACT_TYPE act_f, REGULARIZATION reg) :layer_in(in_size), layer_out(out_size), reg_type(reg), ref_X(nullptr), ref_Z(nullptr) 
	{
		act_func = new Activations(act_f);

		_W.resize(layer_in, layer_out);
		_b.resize(1, layer_out);

		_W.setRandom();  // Temprory Initialization //
		_b.setRandom();

		grads_W.resize(layer_in, layer_out);
		grads_b.resize(1, layer_out);
	}

	virtual ~Dense_FC() {
		delete act_func;
		ref_X = nullptr;
		ref_Z = nullptr;
	}
	
	//virtual bool Forward_Pass(Matrix<float, Dynamic, Dynamic, RowMajor>& _X, Matrix<float, Dynamic, Dynamic, RowMajor>& _A);
	virtual bool Forward_Pass(Tensor4f& _X, Tensor4f& _Z);
	virtual bool Backward_Pass(Tensor4f& dout, Tensor4f& dX);
	virtual void Update_Weights(const float alpha, REGULARIZATION l_type){ std::cout << "\nUpdate Weights\n"; }
	virtual LAYER_TYPE Type() const { return LAYER_TYPE::DENSE; }
	virtual Eigen::VectorXi Out_shape(const Eigen::VectorXi in_shape) {
		Eigen::Vector2i out_shp;
		out_shp << layer_out, 1;  // Hack need to fix ///
		return out_shp;
	}
	virtual void Summary() {
		std::cout << " Dense_FC : " << layer_in << " X " << layer_out << ((reg_type == REGULARIZATION::L1) ? "  L1   P: " : "  L2   P: ") << layer_in * layer_out << '\n';
	}
};

class Max_Pool_2D : public Layer_Base {
private:
	int _FX;
	int _FY;
	int stride;

	// For Backward Pass //
	Tensor4f* ref_X;

public:
	Max_Pool_2D(int f_x = 2, int f_y = 2, int stride_xy = 1) :_FX(f_x), _FY(f_y), stride(stride_xy), ref_X(nullptr){}
	virtual ~Max_Pool_2D() { ref_X = nullptr; }

	virtual bool Forward_Pass(Tensor4f& _X, Tensor4f& _Z);
	virtual bool Backward_Pass(Tensor4f& dout, Tensor4f& dX);
	virtual void Update_Weights(const float alpha, REGULARIZATION l_type) { std::cout << "\nUpdate Weights\n"; }
	virtual LAYER_TYPE Type() const { return LAYER_TYPE::MAX_POOL; }
	virtual Eigen::VectorXi Out_shape(const Eigen::VectorXi in_shape) {
		Eigen::Vector3i out_shp;
		out_shp << in_shape(0), ((in_shape(1) - _FY) / stride + 1), ((in_shape(2) - _FX) / stride + 1);
		return out_shp;
	}
	virtual void Summary() {
		std::cout << " Max_Pool_2D : (" << _FX << " , " << _FY << ")  " << stride << "  P: 0\n";
	}
};

class Convolution : public Layer_Base {
private:
	int _FX;
	int _FY;
	int _FC;
	int num_filters;
	int stride_xy;
	Activations* act_func;
	REGULARIZATION reg_type;

	// Layer Weights and Gradients //
	Tensor4f _W;
	Matrix<float, 1, Dynamic, RowMajor> _b;
	Tensor4f grads_W;
	Matrix<float, 1, Dynamic, RowMajor> grads_b;

	// Refrences for Backward Pass //
	Tensor4f* ref_X;
	Tensor4f* ref_Z;

public:
	Convolution(int f_x, int f_y, int f_c, int n_filter, ACT_TYPE act_f, REGULARIZATION reg) :
		_FX(f_x), _FY(f_y), _FC(f_c), num_filters(n_filter), stride_xy(1), reg_type(reg), ref_X(nullptr), ref_Z(nullptr)
	{
		act_func = new Activations(act_f);

		_W.resize(num_filters, _FC, _FY, _FX);
		_b.resize(1, num_filters);

		_W.setRandom();  // Temprory Initialization //
		_b.setRandom();

		grads_W.resize(num_filters, _FC, _FY, _FX);
		grads_b.resize(1, num_filters);
	}

	virtual ~Convolution() {
		delete act_func;
		ref_Z = nullptr;
		ref_X = nullptr;
	}

	virtual bool Forward_Pass(Tensor4f& _X, Tensor4f& _Z);
	virtual bool Backward_Pass(Tensor4f& dout, Tensor4f& dX);
	virtual void Update_Weights(const float alpha, REGULARIZATION l_type) { std::cout << "\nUpdate Weights\n"; }
	virtual LAYER_TYPE Type() const { return LAYER_TYPE::CONV; }
	virtual Eigen::VectorXi Out_shape(const Eigen::VectorXi in_shape) {
		Eigen::Vector3i out_shp;
		out_shp << num_filters, ((in_shape(1) - _FY) / stride_xy + 1), ((in_shape(2) - _FX) / stride_xy + 1);
		return out_shp;
	}
	virtual void Summary() {
		std::cout << " Conv2D : (" << _FX << ',' << _FY << ',' << _FC << ")  " << stride_xy << "  NF:" << num_filters <<
			((reg_type == REGULARIZATION::L1) ? "  L1   P: " : "  L2   P: ") << _FX * _FY * _FC * num_filters << '\n';
	}
};
