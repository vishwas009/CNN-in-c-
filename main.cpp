#include <iostream>
#include <chrono>
#include "layer.h"
#include "Model.h"

using namespace Eigen;


int main() {
	std::cout << "Testing\n\n";
	//omp_set_num_threads(4);
	
	std::cout << std::boolalpha;
	//Dense_FC fc1(3 * 2 * 3, 9, ACT_TYPE::RELU, REGULARIZATION::L2);
	//Activations act1(ACT_TYPE::SIGMOID);
	
	/*MatrixXf mat(10, 18);
	mat.setRandom();

	Tensor4f t_in(10, 3, 2, 3);
	memcpy(t_in.data(), mat.data(), sizeof(float) * 10 * 18);

	Tensor4f t_out;
	fc1.Forward_Pass(t_in, t_out);
	std::cout << t_out.dimensions() << '\n' << t_out << '\n';

	Tensor4f d_out(10, 1, 3, 3);
	d_out.setRandom();

	Tensor4f d_p;
	std::cout << fc1.Backward_Pass(d_out, d_p) << '\n' << d_p.dimensions() << '\n' << t_out;*/
	
	Tensor4f mX(128, 3, 128, 128);
	mX.setRandom();
	Tensor4f mY(128, 1, 1, 1);
	mY.setRandom();

	Eigen::Vector3i inp_shp;
	inp_shp << 3, 128, 128;

	Model test_model(OPTIMIZER::SGD, LOSS_FUNCTION::MAE, inp_shp);
	test_model.add_Conv2D_Layer(3, 3, 64, ACT_TYPE::RELU, REGULARIZATION::L2);
	test_model.add_Max_Pool_2D_layer(3, 3, 2);
	test_model.add_Conv2D_Layer(3, 3, 32, ACT_TYPE::RELU, REGULARIZATION::L2);
	test_model.add_Max_Pool_2D_layer(3, 3, 2);
	test_model.add_Dense_Layer(64, ACT_TYPE::RELU, REGULARIZATION::L2);
	test_model.add_Dense_Layer(1, ACT_TYPE::SIGMOID, REGULARIZATION::L2);

	std::cout << test_model.Initialize() << '\n';
	test_model.Model_Summary();

	//float res = test_model.model_Forward_Pass(mX, mY);

	/*const int fx = 3, fy = 3, nc = 3;

	Convolution c1(fx, fy, nc, 64, ACT_TYPE::RELU, REGULARIZATION::L2);
	Tensor4f t_in(128, nc, 128, 128);
	t_in.setRandom();

	Tensor4f t_out;

	Tensor4f dout(128, 64, 126, 126);
	dout.setRandom();

	Tensor4f dX;*/

	/*Tensor4f t_in(128, 64, 126, 126);
	t_in.setRandom();

	Tensor4f t_out;

	Max_Pool_2D mp1(3, 3, 2);
	mp1.Forward_Pass(t_in, t_out);

	Tensor4f dout(128, 64, 62, 62);
	dout.setRandom();

	Tensor4f dX;*/

	//float cost = test_model.model_Forward_Pass(mX, mY);

	auto start = std::chrono::high_resolution_clock::now();

	//bool res = mp1.Backward_Pass(dout, dX);
	
	//float cost = test_model.model_Forward_Pass(mX, mY);
	bool res = test_model.Run(mX, mY, 128, 4);
	//res = test_model.Run(mX, mY, 128, 2);
	auto stop = std::chrono::high_resolution_clock::now();
	auto time_taken_1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	/*start = std::chrono::high_resolution_clock::now();
	
	stop = std::chrono::high_resolution_clock::now();
	auto time_taken_2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	
	Eigen::Map< Matrix<float, Dynamic, Dynamic, RowMajor>> t_mat1(t_out.data(), 2000, 2000);
	Eigen::Map< Matrix<float, Dynamic, Dynamic, RowMajor>> t_mat2(t_out2.data(), 2000, 2000);
	std::cout << t_mat1.isApprox(t_mat2);*/

	//std::cout << t_out.dimensions() << '\n';
	//std::cout << dX.dimensions() << '\n';
	std::cout << res << '\n';
	std::cout << "\nTime:  " << time_taken_1.count();
	//std::cout << "\nTime:  " << time_taken_2.count();

	std::cout << "\n\nFinished\n";
	return 0;
}