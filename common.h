#pragma once

#if !defined(NUM_THREADS)
#define NUM_THREADS 4
#endif

#define EIGEN_DONT_PARALLELIZE // Default for now //
// #define EIGEN_USE_MKL_ALL // Performance boost in some operations //
// #define MKL_DIRECT_CALL 1
// #define EIGEN_INITIALIZE_MATRICES_BY_ZERO 1   // Performance Penalties  ///

#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <omp.h>

using namespace Eigen;

typedef Eigen::Tensor<int, 1, Eigen::RowMajor> Tensor1i;
typedef Eigen::Tensor<int, 2, Eigen::RowMajor> Tensor2i;
typedef Eigen::Tensor<int, 3, Eigen::RowMajor> Tensor3i;
typedef Eigen::Tensor<int, 4, Eigen::RowMajor> Tensor4i;
typedef Eigen::Tensor<int, 5, Eigen::RowMajor> Tensor5i;

typedef Eigen::Tensor<float, 1, Eigen::RowMajor> Tensor1f;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<float, 3, Eigen::RowMajor> Tensor3f;
typedef Eigen::Tensor<float, 4, Eigen::RowMajor> Tensor4f;
typedef Eigen::Tensor<float, 5, Eigen::RowMajor> Tensor5f;

typedef Eigen::Tensor<double, 1, Eigen::RowMajor> Tensor1d;
typedef Eigen::Tensor<double, 2, Eigen::RowMajor> Tensor2d;
typedef Eigen::Tensor<double, 3, Eigen::RowMajor> Tensor3d;
typedef Eigen::Tensor<double, 4, Eigen::RowMajor> Tensor4d;
typedef Eigen::Tensor<double, 5, Eigen::RowMajor> Tensor5d;

struct _XY {
	int x;
	int y;
};

struct _XYZ {
	int x;
	int y;
	int z;
};

enum class ACT_TYPE : uint8_t {
	LINEAR = 1,
	RELU,
	LEAKY_RELU,
	SIGMOID,
	SOFTMAX,
	TANH
};

enum class LAYER_TYPE : uint8_t { // To be Expanded //
	DENSE = 1,
	CONV,
	MAX_POOL,
	AVG_POOL
};

enum class REGULARIZATION : uint8_t {
	L1 = 1,
	L2
};

enum class OPTIMIZER : uint8_t {
	SGD = 1,
	SGD_MOMENTUM,
	SGD_STOCHASTIC,
	ADAM
};

enum class LOSS_FUNCTION : uint8_t {
	MSE = 0,
	MAE,
	BINARY_CROSS_ENTROPY,
	CATEGORICAL_CROSS_ENTROPY
};




