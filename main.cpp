#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include "cnn.h"
//#include "p_gfx.h"



int main(void) {
	std::cout << "Testing\n\n";
	omp_set_num_threads(4);

	const int m_size = 10240;

	///// Loading Labels from attrib file ///////////////////////////

	std::ifstream attrib("d_list_128_labels.txt");
	if (!attrib.is_open()) { std::cout << "Error Attrib File not opened\n"; return -1; }

	char buff[15] = { 0 };
	Matrix<float> y_labels({ m_size ,1});

	for (int i = 0; i < m_size; i++) {
		attrib.getline(buff, 14);

		//if (buff[0] == 'f' || buff[0] == 'm')y_labels(i) = 1;
		if (buff[0] == 'f')y_labels(i) = 1;
		if (buff[0] == 'o')y_labels(i) = 0;
	}

	int nf = 0, no = 0;
	for (int i = 0; i < m_size; i++) {
		if (y_labels(i) == 1)nf++;
		if (y_labels(i) == 0)no++;
	}

	std::cout << "\ny_labels loaded. \nNo. of Face Examples =  " << nf << "   No. of Random Objects =  " << no << '\n';

	const int img_ht = 128;
	const int img_wd = 128;
	const int img_chn = 3;
	const int num_labels = 1;

	std::cout << "\nLoading First Batch\n";

	const int batch_size = 128;
	const int n_batches = m_size / batch_size;

	Matrix<float> mX[n_batches];
	Matrix<float> mY[n_batches];

	/*FILE* data_file = fopen("F:/MISC/ProG_ProJ/_dst/10^5_dataset.dat", "rb");
	bgra8* tmp_buff = new bgra8[109 * 89];

	for (int n = 0; n < n_batches; n++) {
		mX[n].resize({ batch_size,img_chn,img_ht,img_wd });
		mY[n].resize({ batch_size,num_labels });
		for (int m = 0; m < batch_size; m++) {
			fread(tmp_buff, sizeof(bgra8), img_ht * img_wd, data_file);
			for (int i = 0; i < img_ht; i++)
				for (int j = 0; j < img_wd; j++) {
					mX[n](m, 0, i, j) = (float)(tmp_buff[i * img_wd + j].r) / 255.0f;
					mX[n](m, 1, i, j) = (float)(tmp_buff[i * img_wd + j].g) / 255.0f;
					mX[n](m, 2, i, j) = (float)(tmp_buff[i * img_wd + j].b) / 255.0f;
				}

			mY[n](m, 0) = y_labels(n * batch_size + m);
		}
	}*/

	FILE* path_list = fopen("d_list_128.txt", "r");
	char line[100] = { 0 };
	char* cntxt = 0;
	bgra8* tmp_buff = new bgra8[img_ht * img_wd];

	for (int n = 0; n < n_batches; n++) {
		mX[n].resize({ batch_size,img_chn,img_ht,img_wd });
		mY[n].resize({ batch_size,num_labels });
		for (int m = 0; m < batch_size; m++) {
			fgets(line, 64, path_list);
			strtok_s(line, "\n", &cntxt);
			if (!load_img_data(line, tmp_buff)) { std::cerr << "Error Loading Image\n"; return -1; }
			for (int i = 0; i < img_ht; i++)
				for (int j = 0; j < img_wd; j++) {
					mX[n](m, 0, i, j) = (float)(tmp_buff[i * img_wd + j].r) / 255.0f;
					mX[n](m, 1, i, j) = (float)(tmp_buff[i * img_wd + j].g) / 255.0f;
					mX[n](m, 2, i, j) = (float)(tmp_buff[i * img_wd + j].b) / 255.0f;
				}

			mY[n](m, 0) = y_labels(n * batch_size + m);
		}
	}

	std::cout << "\nFirst Batch Loaded in Matrices\n";
	
	

	Shape w_mat_shapes[4];
	w_mat_shapes[0] = { 64,img_chn,3,3 };
	w_mat_shapes[1] = { 32,64,3,3 };
	w_mat_shapes[2] = { 28800,64 };
	w_mat_shapes[3] = { 64,num_labels };

	Parameters<float> theta(w_mat_shapes, 4);
	theta.init_Zero();
	He_Initialization(theta);

	theta.b[0].resize({ 64,1 });
	theta.b[0].set_Zero();
	theta.b[1].resize({ 32,1 });
	theta.b[1].set_Zero();
	theta.b[2].resize({ 64,1 });
	theta.b[2].set_Zero();

	theta.b[3].resize({ num_labels,2 }); theta.b[3].resize({ num_labels,1 }); /// Hack need to Fix ////
	theta.b[3].set_Zero();
	
	std::cout << "\nWeight Matrices Initialized\n";

	Parameters<float> theta_grads(w_mat_shapes, 4);
	theta_grads.init_Zero();

	theta_grads.b[0].resize({ 64,1 });
	theta_grads.b[0].set_Zero();
	theta_grads.b[1].resize({ 32,1 });
	theta_grads.b[1].set_Zero();
	theta_grads.b[2].resize({ 64,1 });
	theta_grads.b[2].set_Zero();

	theta_grads.b[3].resize({ num_labels,2 }); theta_grads.b[3].resize({ num_labels,1 }); /// Hack need to Fix ////
	theta_grads.b[3].set_Zero();

	Parameters<float> v_grads(w_mat_shapes, 4);
	v_grads.init_Zero();

	v_grads.b[0].resize({ 64,1 });
	v_grads.b[0].set_Zero();
	v_grads.b[1].resize({ 32,1 });
	v_grads.b[1].set_Zero();
	v_grads.b[2].resize({ 64,1 });
	v_grads.b[2].set_Zero();

	v_grads.b[3].resize({ num_labels,2 }); v_grads.b[3].resize({ num_labels,1 }); /// Hack need to Fix ////
	v_grads.b[3].set_Zero();

	auto start = std::chrono::high_resolution_clock::now();
	
	Matrix hx = Backward_Pass(mX[0], mY[0], theta, theta_grads);
	//Matrix hx = Forward_Pass(mX[0], theta);

	auto stop = std::chrono::high_resolution_clock::now();
	auto time_taken_1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "\nTime:  " << time_taken_1.count() << "\n\n";
	//hx.shape().print();
	
	float cost = Cross_Entropy_Cost(hx, mY[0]);
	std::cout << "\nFirst Batch Initial Cost:  " << cost << '\n';
	//return 0;

	const float alpha = 0.005, lambda = 0.25, beta = 0.9;
	float tr_pos = 0;
	float pred = 0;
	for (int i = 0; i < batch_size; i++) {
		pred = (hx(i, 0) > 0.80);
		if (pred == mY[0](i, 0))tr_pos++;
	}

	std::cout << "\nInitial Accuracy: " << (tr_pos / batch_size) * 100 << '\n';

	for (int epochs = 0; epochs < 1; epochs++) {

		std::cout << "\nEpoch  " << epochs + 1 << '\n';
		for (int i = 0; i < n_batches; i++) {

			hx = Backward_Pass(mX[i], mY[i], theta, theta_grads, lambda);

			cost = Cross_Entropy_Cost(hx, mY[i]);
			if (lambda > 0) {
				float reg = 0;
				for (int r = 0; r < 4; r++)
					reg += theta.W[r].square().sum();

				reg = (reg * lambda) / (batch_size * 2);
				cost += reg;
			}

			for (int k = 0; k < 4; k++) {
				v_grads.W[k] = beta * v_grads.W[k] + (1 - beta) * theta_grads.W[k];
				v_grads.b[k] = beta * v_grads.b[k] + (1 - beta) * theta_grads.b[k];

				theta.W[k] = theta.W[k] - v_grads.W[k] * alpha;
				theta.b[k] = theta.b[k] - v_grads.b[k] * alpha;
			}

			std::cout << "Mini Batch Cost:  " << cost << '\n';
		}

		if (epochs < 1 - 1) {
			std::cout << "\nLoading Batch \n";
			for (int n = 0; n < n_batches; n++) {
				for (int m = 0; m < batch_size; m++) {
					fgets(line, 64, path_list);
					strtok_s(line, "\n", &cntxt);
					if (!load_img_data(line, tmp_buff)) { std::cerr << "Error Loading Image\n"; return -1; }
					for (int i = 0; i < img_ht; i++)
						for (int j = 0; j < img_wd; j++) {
							mX[n](m, 0, i, j) = (float)(tmp_buff[i * img_wd + j].r) / 255.0f;
							mX[n](m, 1, i, j) = (float)(tmp_buff[i * img_wd + j].g) / 255.0f;
							mX[n](m, 2, i, j) = (float)(tmp_buff[i * img_wd + j].b) / 255.0f;
						}

					attrib.getline(buff, 14);

					if (buff[0] == 'f')mY[n](m, 0) = 1;
					if (buff[0] == 'o')mY[n](m, 0) = 0;

				}
			}
			std::cout << "Batch Loaded\n";
		}

	}

	tr_pos = 0; pred = 0;
	for (int i = 0; i < batch_size; i++) {
		pred = (hx(i, 0) > 0.80);
		if (pred == mY[n_batches-1](i, 0))tr_pos++;
	}

	std::cout << "\nAccuracy: " << (tr_pos / (float)batch_size) * 100.0 << '\n';
	//hx.print();
	//mY[1].print();

	
	/*std::ofstream _W[4];
	std::ofstream _B[4];
	std::string wfiles = "W_";
	std::string bfiles = "b_";

	for (int n = 0; n < 4; n++) {
		wfiles = "W_";
		wfiles = wfiles + std::to_string(n + 1);
		_W[n].open(wfiles, std::ios::binary | std::ios::app);
		_W[n].write((char*)theta.W[n].data(), sizeof(float) * theta.W[n].size());
		_W[n].close();

		bfiles = "b_";
		bfiles = bfiles + std::to_string(n + 1);
		_B[n].open(bfiles, std::ios::binary | std::ios::app);
		_B[n].write((char*)theta.b[n].data(), sizeof(float) * theta.b[n].size());
		_B[n].close();
	}*/


	std::cout << "\n\nFinished\n";
	return 0;
}