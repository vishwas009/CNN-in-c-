#pragma once
#include "common.h"

class Activation_Base {
public:
	Activation_Base() {}
	virtual ~Activation_Base() {};

	virtual void compute_Forward_Inp(float* z_ptr, const size_t n) = 0;
	virtual void compute_Forward(const float* z_ptr, float* a_ptr, const size_t n) = 0;
	virtual void compute_Gradient(const float* z_ptr, float* out_ptr, const size_t n) = 0;
	virtual void compute_Backprop_Inp(float* z_ptr, const size_t n) = 0;
};

class Relu : public Activation_Base {
private:
	bool junk;

public:
	Relu() :junk(true) {}
	virtual ~Relu() {}

	virtual void compute_Forward(const float* z_ptr, float* a_ptr, const size_t n) {
		for (int i = 0; i < n; i++)
			a_ptr[i] = (z_ptr[i] > 0) ? z_ptr[i] : 0;
	}

	virtual void compute_Forward_Inp(float* z_ptr, const size_t n) {
		// ToDo Optimize ///
		/*__m256 zero = _mm256_setzero_ps();
		int i = 0;
		for (i = 0; i <= n - 8; i += 8)
			_mm256_store_ps(z_ptr + i, _mm256_max_ps(_mm256_load_ps(z_ptr + i), zero));

		for (; i < n; i++)
			z_ptr[i] = (z_ptr[i] > 0) ? z_ptr[i] : 0;*/
#pragma omp parallel for
		for (int i = 0; i < n; i++)
			z_ptr[i] = (z_ptr[i] > 0) ? z_ptr[i] : 0;
	}

	virtual void compute_Gradient(const float* z_ptr, float* out_ptr, const size_t n) {
		// ToDo Optimize ///
		for (int i = 0; i < n; i++)
			out_ptr[i] = (z_ptr[i] >= 0);
	}

	virtual void compute_Backprop_Inp(float* z_ptr, const size_t n) {
		for (int i = 0; i < n; i++)
			z_ptr[i] = (z_ptr[i] != 0);
	}

};

class Leaky_Relu : public Activation_Base {
private:
	bool junk;

public:
	Leaky_Relu() :junk(true) {}
	virtual ~Leaky_Relu() {}

	virtual void compute_Forward(const float* z_ptr, float* a_ptr, const size_t n) {
		// ToDo Optimize ///
		for (int i = 0; i < n; i++)
			a_ptr[i] = (z_ptr[i] >= 0) ? z_ptr[i] : 0.01f * z_ptr[i];
	}

	virtual void compute_Forward_Inp(float* z_ptr, const size_t n) {
		// ToDo Optimize ///
		for (int i = 0; i < n; i++)
			z_ptr[i] = (z_ptr[i] >= 0) ? z_ptr[i] : 0.01f * z_ptr[i];
	}

	virtual void compute_Gradient(const float* z_ptr, float* out_ptr, const size_t n) {
		// ToDo Optimize ///
		for (int i = 0; i < n; i++)
			out_ptr[i] = (z_ptr[i] >= 0) ? 1 : 0.01f;
	}

	virtual void compute_Backprop_Inp(float* z_ptr, const size_t n) {
		for (int i = 0; i < n; i++)
			z_ptr[i] = (z_ptr[i] >= 0) ? 1 : 0.01f;
	}

};

class Sigmoid : public Activation_Base {
private:
	bool junk;

public:
	Sigmoid() :junk(true) {}
	virtual ~Sigmoid() {}

	virtual void compute_Forward(const float* z_ptr, float* a_ptr, const size_t n) {
		const Eigen::TensorMap<Tensor2f> z_t(const_cast<float*>(z_ptr), 1, n);
		Eigen::TensorMap<Tensor2f> a_t(a_ptr, 1, n);
		a_t = z_t.sigmoid();
	}

	virtual void compute_Forward_Inp(float* z_ptr, const size_t n) {
		// ToDo Optimize  Strongly!!!! ///
		for (int i = 0; i < n; i++)
			z_ptr[i] = 1.0 / (1.0 + std::expf(-z_ptr[i]));
	}

	virtual void compute_Gradient(const float* z_ptr, float* out_ptr, const size_t n) {
		// ToDo Optimize  Strongly!!!! ///
		float tmp = 0;
		for (int i = 0; i < n; i++) {
			tmp = 1.0 / (1.0 + std::expf(-z_ptr[i]));
			out_ptr[i] = tmp * (1.0 - tmp);
		}
	}

	virtual void compute_Backprop_Inp(float* z_ptr, const size_t n) {
		for (int i = 0; i < n; i++)
			z_ptr[i] = z_ptr[i] * (1.0 - z_ptr[i]);
	}

};

class Activations {
private:
	Activation_Base* act_ptr;
	ACT_TYPE act_func_type;



public:
	Activations(ACT_TYPE act_func) :act_ptr(nullptr), act_func_type(act_func) {
		switch (act_func) {
		case ACT_TYPE::RELU: { act_ptr = new Relu(); return; }
		case ACT_TYPE::LEAKY_RELU: { act_ptr = new Leaky_Relu(); return; }
		case ACT_TYPE::SIGMOID: { act_ptr = new Sigmoid(); return; }
		default: {act_ptr = nullptr; return; }
		}
	}

	virtual ~Activations() { delete act_ptr; }

	Tensor1f Forward(const Tensor1f& t) {
		Tensor1f tmp; tmp.resize(t.dimensions());
		act_ptr->compute_Forward(t.data(), tmp.data(), t.size());
		return tmp;
	}

	Tensor2f Forward(const Tensor2f& t) {
		Tensor2f tmp; tmp.resize(t.dimensions());
		act_ptr->compute_Forward(t.data(), tmp.data(), t.size());
		return tmp;
	}

	Tensor3f Forward(const Tensor3f& t) {
		Tensor3f tmp; tmp.resize(t.dimensions());
		act_ptr->compute_Forward(t.data(), tmp.data(), t.size());
		return tmp;
	}

	Tensor4f Forward(const Tensor4f& t) {
		Tensor4f tmp; tmp.resize(t.dimensions());
		act_ptr->compute_Forward(t.data(), tmp.data(), t.size());
		return tmp;
	}

	void Forward_Inp(Tensor2f& t) const { return act_ptr->compute_Forward_Inp(t.data(), t.size()); }
	void Forward_Inp(Tensor3f& t) const { return act_ptr->compute_Forward_Inp(t.data(), t.size()); }
	void Forward_Inp(Tensor1f& t) const { return act_ptr->compute_Forward_Inp(t.data(), t.size()); }
	void Forward_Inp(Tensor4f& t) const { return act_ptr->compute_Forward_Inp(t.data(), t.size()); }
	void Forward_Inp(Matrix<float, Dynamic, Dynamic, RowMajor>& mat) { return act_ptr->compute_Forward_Inp(mat.data(), mat.size()); }
	//void Forward_Inp(float* z_ptr, int size) const { return act_ptr->compute_Forward_Inp(z_ptr, size); }

	void Backprop_Inp(Tensor2f& t) const { return act_ptr->compute_Backprop_Inp(t.data(), t.size()); }
	void Backprop_Inp(Tensor3f& t) const { return act_ptr->compute_Backprop_Inp(t.data(), t.size()); }
	void Backprop_Inp(Tensor1f& t) const { return act_ptr->compute_Backprop_Inp(t.data(), t.size()); }
	void Backprop_Inp(Tensor4f& t) const { return act_ptr->compute_Backprop_Inp(t.data(), t.size()); }

	Tensor1f Gradient(const Tensor1f& t_in) const {
		Tensor1f tmp; tmp.resize(t_in.dimensions());
		act_ptr->compute_Gradient(t_in.data(), tmp.data(), t_in.size());
		return tmp;
	}

	Tensor2f Gradient(const Tensor2f& t_in) const {
		Tensor2f tmp; tmp.resize(t_in.dimensions());
		act_ptr->compute_Gradient(t_in.data(), tmp.data(), t_in.size());
		return tmp;
	}

	Tensor3f Gradient(const Tensor3f& t_in) const {
		Tensor3f tmp; tmp.resize(t_in.dimensions());
		act_ptr->compute_Gradient(t_in.data(), tmp.data(), t_in.size());
		return tmp;
	}

	Tensor4f Gradient(const Tensor4f& t_in) const {
		Tensor4f tmp; tmp.resize(t_in.dimensions());
		act_ptr->compute_Gradient(t_in.data(), tmp.data(), t_in.size());
		return tmp;
	}

};