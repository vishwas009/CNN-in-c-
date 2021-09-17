#pragma once
#include "nn_matrix.h"
#include <chrono>
#include <future>
//#include "p_gfx.h"
#include <jpeglib.h>

enum class L_Type : uint8_t {
	CONVOLUTION = 1,
	DENSE
};

struct bgra8 {
	unsigned char b = 0;
	unsigned char g = 0;
	unsigned char r = 0;
	unsigned char a = 0;
};

template<typename T>
struct Layer_Theta {
	Matrix<T> W;
	Matrix<T> b;
	L_Type l_type;
};

template<typename T>
struct Parameters {
	int n_matrices;
	Shape* w_shapes = nullptr;
	Matrix<T>* W = nullptr;
	Matrix<T>* b = nullptr;

	Parameters(const Shape* shapes, int n_mats) {
		w_shapes = new Shape[n_mats];
		n_matrices = n_mats;
		
		for (int i = 0; i < n_mats; i++)
			w_shapes[i] = shapes[i];

		W = new Matrix<T>[n_mats];
		b = new Matrix<T>[n_mats];
	}

	void init_Zero() {
		for (int i = 0; i < n_matrices; i++) {
			W[i].resize(w_shapes[i]);
			W[i].set_Zero();
		}
	}

	~Parameters() {
		delete[] W;
		delete[] b;
		delete[] w_shapes;
	}
};

bool load_img_data(char* path, bgra8* img_buff);

bool Convolve_Forward(const Matrix<float>& a_prev, const Matrix<float>& w_conv, const Matrix<float>& bias, Matrix<float>& output, const int stride = 1);

bool Convolve_Forward_v2(const Matrix<float>& a_prev, const Matrix<float>& w_conv, const Matrix<float>& bias, Matrix<float>& output, const int stride = 1);

bool Convolve_Backward(const Matrix<float>& aX, const Matrix<float>& dout, const Matrix<float>& w_conv, Matrix<float>& w_grad, Matrix<float>& b_grad, Matrix<float>& dX, const int stride = 1);

bool Convolve_Backward_v2(const Matrix<float>& aX, const Matrix<float>& dout, const Matrix<float>& w_conv, Matrix<float>& w_grad, Matrix<float>& b_grad, Matrix<float>& dX, const int stride = 1);

bool Convolve_Backward_v3(const Matrix<float>& aX, const Matrix<float>& dout, const Matrix<float>& w_conv, Matrix<float>& w_grad, Matrix<float>& b_grad, Matrix<float>& dX, const int stride = 1);

Matrix<float> Max_Pool_Forward(const Matrix<float>& a_prev, const int f, const int stride);

Matrix<float> Max_Pool_Backward(const Matrix<float>& aX, Matrix<float>& dout, const int f, const int stride);

Matrix<float> Relu(const Matrix<float>& z);

void Relu_Inp(Matrix<float>& z);

Matrix<float> Relu_Gradient(const Matrix<float>& z);

void Relu_Gradient_v2(Matrix<float>& z);

Matrix<float> Sigmoid(const Matrix<float>& z);

void Sigmoid_Inp(Matrix<float>& z);

Matrix<float> Sigmoid_Gradient(const Matrix<float>& z);

void Sigmoid_Gradient_v2(Matrix<float>& z);

float Cross_Entropy_Cost(Matrix<float>& hx, Matrix<float>& Y);

void He_Initialization(Parameters<float>& theta);

Matrix<float> Forward_Pass(const Matrix<float>& X, const Parameters<float>& weights);

Matrix<float> Backward_Pass(const Matrix<float>& X, Matrix<float>& Y, const Parameters<float>& weights, Parameters<float>& weights_grads, const float lambda = 0.0);