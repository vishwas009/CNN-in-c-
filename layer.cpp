#include <future>
#include "layer.h"

static void convolve_2d_batch_st(const float* _X, float* _W, const float* _b, float* _Z, const int batch_size, const int ht, const int wd,
	const int x_nc, const int w_nf, const int _FX = 3, const int _FY = 3, const int stride = 1) 
{
	const int new_nc = w_nf;
	const int new_wd = (wd - _FX) / stride + 1;
	const int new_ht = (ht - _FY) / stride + 1;

	Eigen::Map<Matrix<float, Dynamic, Dynamic, RowMajor>> mat_W(_W, w_nf, x_nc * _FX * _FY);
	Matrix<float, Dynamic, Dynamic, RowMajor> mat_X_col;
	mat_X_col.resize(x_nc * _FY * _FX, new_ht * new_wd);
	float* mat_X_col_ptr = mat_X_col.data();

	//float* mat_X_col_ptr = (float*)mkl_malloc(sizeof(float) * x_nc * _FY * _FX * new_ht * new_wd, 64);

	for (int n = 0; n < batch_size; n++) {
		const int ap_exm_ofst = n * x_nc * ht * wd;
		const int ac_chn_ofst = new_ht * new_wd;
		const int num_el = sizeof(float) * new_wd;
		for (int y = 0; y < new_ht; y++)
			for (int z = 0; z < x_nc; z++)
				for (int fy = 0; fy < _FY; fy++)
					for (int fx = 0; fx < _FX; fx++)
						memcpy(&mat_X_col_ptr[(z * _FY * _FX + fy * _FY + fx) * ac_chn_ofst + y * new_wd], &_X[ap_exm_ofst + z * ht * wd + (y + fy) * wd + fx], num_el);

		Eigen::Map<Matrix<float, Dynamic, Dynamic, RowMajor>> mat_Z(_Z + n * new_nc * new_ht * new_wd, w_nf, new_ht * new_wd);
		
		const int row_len = new_ht * new_wd;
		const int az_exm_ofst = n * new_nc * new_ht * new_wd;
		for (int z = 0; z < new_nc; z++) {
			__m256 bias = _mm256_set1_ps(_b[z]);
			int i = 0;
			for (i = 0; i <= row_len - 8; i += 8)
				_mm256_store_ps(&_Z[az_exm_ofst + z * row_len + i], bias);

			for (int rmd = i; rmd < row_len; rmd++)
				_Z[az_exm_ofst + z * row_len + rmd] = _b[z];
		}

		mat_Z.noalias() += mat_W * mat_X_col;

		///*cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		//	new_nc, new_ht * new_wd, x_nc * _FY * _FX,
		//	1.0, _W, x_nc * _FY * _FX, mat_X_col_ptr, new_ht * new_wd,
		//	0.0, _Z + n * new_nc * new_ht * new_wd, new_ht * new_wd);*/

		/*const int row_len = new_ht * new_wd;
		const int az_exm_ofst = n * new_nc * new_ht * new_wd;
		for (int z = 0; z < new_nc; z++) {
			__m256 bias = _mm256_set1_ps(_b[z]);
			int i = 0;
			for (i = 0; i <= row_len - 8; i += 8)
				_mm256_store_ps(&_Z[az_exm_ofst + z * row_len + i], _mm256_add_ps(_mm256_load_ps(&_Z[az_exm_ofst + z * row_len + i]), bias));

			for (int rmd = i; rmd < row_len; rmd++)
				_Z[az_exm_ofst + z * row_len + rmd] += _b[z];
		}*/

	}

	//mkl_free(mat_X_col_ptr);
}

static void convolve_back_batch_st(const float* _X, float* dout, float* _W, float* grad_W, float* grad_b, float* dX, const int batch_size,
	int a_nc, int a_ht, int a_wd, int d_nc, int d_ht, int d_wd, int _FX, int _FY, int _FC, int stride = 1)
{
	const int n_wd = (a_wd - _FX) / stride + 1;
	const int n_ht = (a_ht - _FY) / stride + 1;
	const int w_nf = d_nc;

	Matrix<float, Dynamic, Dynamic, RowMajor> _X_col_T(d_ht * d_wd, a_nc * _FY * _FX);
	Matrix<float, Dynamic, Dynamic, RowMajor> dX_col(a_nc * _FY * _FX, d_ht * d_wd);
	const Matrix<float, Dynamic, Dynamic, RowMajor> mat_W_T = Eigen::Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(_W, w_nf, _FC * _FX * _FY).transpose();
	Eigen::Map<Matrix<float, Dynamic, Dynamic, RowMajor>> grad_W_acc(grad_W, w_nf, _FC * _FX * _FY);
	Eigen::Map<Matrix<float, 1, Dynamic, RowMajor>> grad_b_acc(grad_b, 1, w_nf);
	grad_W_acc.setZero();
	grad_b_acc.setZero();

	float* _X_col_T_ptr = _X_col_T.data();
	float* dX_col_ptr = dX_col.data();

	for (int n = 0; n < batch_size; n++) {
		const int ap_exm_ofst = n * a_nc * a_ht * a_wd;
		const int ac_exm_ofst = a_nc * _FY * _FX;
		for (int y = 0; y < d_ht; y++)
			for (int x = 0; x < d_wd; x++)
				for (int z = 0; z < a_nc; z++)
					for (int fy = 0; fy < _FY; fy++)
						for (int fx = 0; fx < _FX; fx++)
							_X_col_T_ptr[(y * d_wd + x) * ac_exm_ofst + z * _FY * _FX + fy * _FX + fx] = _X[ap_exm_ofst + z * a_ht * a_wd + (y + fy) * a_wd + fx + x];

		Eigen::Map<Matrix<float, Dynamic, Dynamic, RowMajor>> mat_dout(dout + n * d_nc * d_ht * d_wd, d_nc, d_ht * d_wd); // dZL //

		grad_W_acc.noalias() += mat_dout * _X_col_T;

		dX_col.noalias() = mat_W_T * mat_dout;

		int dx_chn_ofst = 0;
		int dxc_chn_ofst = 0;
		int dx_exm_ofst = 0;
		int dxc_img_ofst = 0;
		for (int z = 0; z < a_nc; z++) {
			dx_chn_ofst = z * a_ht * a_wd;
			dxc_chn_ofst = z * _FY * _FX;
			for (int fy = 0; fy < _FY; fy++) {
				for (int fx = 0; fx < _FX; fx++) {
					dxc_img_ofst = d_ht * d_wd;
					dx_exm_ofst = n * a_nc * a_ht * a_wd;
					for (int y = 0; y < d_ht; y++) {
						int x = 0;
						for (x = 0; x <= d_wd - 8; x += 8)
							_mm256_store_ps(&dX[dx_exm_ofst + dx_chn_ofst + (y + fy) * a_wd + x + fx],
								_mm256_add_ps(_mm256_load_ps(&dX[dx_exm_ofst + dx_chn_ofst + (y + fy) * a_wd + x + fx]),
									_mm256_load_ps(&dX_col_ptr[(dxc_chn_ofst + fy * _FX + fx) * dxc_img_ofst + y * d_wd + x])));

						for (; x < d_wd; x++)
							dX[dx_exm_ofst + dx_chn_ofst + (y + fy) * a_wd + x + fx] += dX_col_ptr[(dxc_chn_ofst + fy * _FX + fx) * dxc_img_ofst + y * d_wd + x];

					}
				}
			}
		}

		grad_b_acc.noalias() += mat_dout.rowwise().sum();
	}

}

bool Dense_FC::Forward_Pass(Tensor4f& _X, Tensor4f& _Z)
{
	if (_X.dimension(1) * _X.dimension(2) * _X.dimension(3) != layer_in)return false;
	ref_X = &_X;  // Reference for Backward Pass//

	Eigen::Map< Matrix<float, Dynamic, Dynamic, RowMajor>> mat_X(_X.data(), _X.dimension(0), _X.dimension(1) * _X.dimension(2) * _X.dimension(3));
	_Z.resize(_X.dimension(0), layer_out, 1, 1);

	Eigen::Map< Matrix<float, Dynamic, Dynamic, RowMajor>> mat_Z(_Z.data(), _X.dimension(0), layer_out);
	mat_Z.noalias() = (mat_X * _W).rowwise() + _b;

	ref_Z = &_Z;
	act_func->Forward_Inp(_Z);

	return true;
}

bool Dense_FC::Backward_Pass(Tensor4f& dout, Tensor4f& dX)
{
	if (dout.dimension(1) * dout.dimension(2) * dout.dimension(3) != layer_out)return false;
	if (dout.dimension(0) != ref_X->dimension(0))return false;
	const int m = dout.dimension(0);
	act_func->Backprop_Inp(*ref_Z); 
	dout *= *ref_Z;
	Eigen::Map< Matrix<float, Dynamic, Dynamic, RowMajor>> mat_dout(dout.data(), dout.dimension(0), dout.dimension(1) * dout.dimension(2) * dout.dimension(3));  // shape(m,layer_out)//
	Eigen::Map< Matrix<float, Dynamic, Dynamic, RowMajor>> mat_X(ref_X->data(), ref_X->dimension(0), ref_X->dimension(1) * ref_X->dimension(2) * ref_X->dimension(3)); // shape(m,layer_in)//

	grads_W.noalias() = (1.0f / m) * (mat_X.transpose() * mat_dout);  // shape(layer_in, layer_out)  Regularization not performed yet on Gradients //
	grads_b.noalias() = (1.0f / m) * (mat_dout.colwise().sum());  // shape(1, layer_out) //

	dX.resize(m, layer_in, 1, 1);
	Eigen::Map< Matrix<float, Dynamic, Dynamic, RowMajor>> mat_dX(dX.data(), m, layer_in); 
	mat_dX.noalias() = mat_dout * _W.transpose(); // shape(m, layer_in) //

	return true;
}

bool Max_Pool_2D::Forward_Pass(Tensor4f& _X, Tensor4f& _Z)
{
	// ToDo error checking //
	ref_X = &_X; // Refrence for Backward Pass //
	const int m = _X.dimension(0);
	const int a_nc = _X.dimension(1);
	const int i_ht = _X.dimension(2);
	const int i_wd = _X.dimension(3);
	
	const int new_wd = (i_wd - _FX) / stride + 1;
	const int new_ht = (i_ht - _FY) / stride + 1;

	_Z.resize(m, a_nc, new_ht, new_wd);
	const float* ap_ptr = _X.data();
	float* out_ptr = _Z.data();

#pragma omp parallel for
	for (int n = 0; n < m; n++) {
		float t_max = 0;
		int ap_exm_ofst = n * a_nc * i_ht * i_wd;
		int out_exm_ofst = n * a_nc * new_ht * new_wd;
		for (int z = 0; z < a_nc; z++) {
			int ap_ofst = ap_exm_ofst + z * i_ht * i_wd;
			int out_ofst = out_exm_ofst + z * new_ht * new_wd;
			for (int y = 0; y < new_ht; y++)
				for (int x = 0; x < new_wd; x++) {
					t_max = ap_ptr[ap_ofst + (y * stride) * i_wd + x * stride];
					for (int fy = 0; fy < _FY; fy++)
						for (int fx = 0; fx < _FX; fx++)
							if (t_max < ap_ptr[ap_ofst + (y * stride + fy) * i_wd + x * stride + fx])
								t_max = ap_ptr[ap_ofst + (y * stride + fy) * i_wd + x * stride + fx];
					out_ptr[out_ofst + y * new_wd + x] = t_max;
				}

		}
	}

	return true;
}

bool Max_Pool_2D::Backward_Pass(Tensor4f& dout, Tensor4f& dX)
{
	const int m = ref_X->dimension(0);
	const int a_nc = ref_X->dimension(1);
	const int i_ht = ref_X->dimension(2);
	const int i_wd = ref_X->dimension(3);
	const int d_wd = (i_wd - _FX) / stride + 1;
	const int d_ht = (i_ht - _FY) / stride + 1;

	if (dout.dimension(0) != m)return false;
	if (dout.dimension(1) * dout.dimension(2) * dout.dimension(3) != a_nc * d_ht * d_wd)return false;

	Eigen::array<__int64, 4> dim4({ m,a_nc,d_ht,d_wd });
	dout = dout.reshape(dim4);

	dX.resize(m, a_nc, i_ht, i_wd); // shape(ref_X.dimensions()) //
	dX.setZero();

	const float* _X_ptr = ref_X->data();
	const float* dout_ptr = dout.data();
	float* dX_ptr = dX.data();

#pragma omp parallel for
	for (int n = 0; n < m; n++) {
		float t_max = 0;
		int pos = 0;
		int aX_exm_ofst = n * a_nc * i_ht * i_wd;
		int dout_exm_ofst = n * a_nc * d_ht * d_wd;
		for (int z = 0; z < a_nc; z++) {
			int aX_ofst = aX_exm_ofst + z * i_ht * i_wd;
			int dout_ofst = dout_exm_ofst + z * d_ht * d_wd;
			for (int y = 0; y < d_ht; y++)
				for (int x = 0; x < d_wd; x++) {
					pos = aX_ofst + (y * stride) * i_wd + x * stride;
					t_max = _X_ptr[pos];
					for (int fy = 0; fy < _FY; fy++)
						for (int fx = 0; fx < _FX; fx++)
							if (t_max < _X_ptr[aX_ofst + (y * stride + fy) * i_wd + x * stride + fx]) {
								pos = aX_ofst + (y * stride + fy) * i_wd + x * stride + fx;
								t_max = _X_ptr[pos];
							}
					dX_ptr[pos] += dout_ptr[dout_ofst + y * d_wd + x];

				}
		}
	}

	return true;
}

bool Convolution::Forward_Pass(Tensor4f& _X, Tensor4f& _Z)
{
	//ToDo error checking //
	const int m = _X.dimension(0);
	const int a_nc = _X.dimension(1);
	const int i_ht = _X.dimension(2);
	const int i_wd = _X.dimension(3);
	const int new_wd = (i_wd - _FX) / stride_xy + 1;
	const int new_ht = (i_ht - _FY) / stride_xy + 1;
	const int new_nc = num_filters;

	if (a_nc != _FC) return false;
	ref_X = &_X;
	_Z.resize(m, new_nc, new_ht, new_wd);
	
#if defined(EIGEN_DONT_PARALLELIZE)

	int batch_size = m / NUM_THREADS;
	{
		std::future<void> conv_async_thds[NUM_THREADS];

		for (int i = 0; i < NUM_THREADS - 1; i++)
			conv_async_thds[i] = std::async(std::launch::async, convolve_2d_batch_st,
				_X.data() + i * batch_size * a_nc * i_ht * i_wd, _W.data(), _b.data(), _Z.data() + i * batch_size * new_nc * new_ht * new_wd,
				batch_size, i_ht, i_wd, a_nc, num_filters, _FX, _FY, stride_xy);

		conv_async_thds[NUM_THREADS - 1] = std::async(std::launch::async, convolve_2d_batch_st,
			_X.data() + (NUM_THREADS - 1) * batch_size * a_nc * i_ht * i_wd, _W.data(), _b.data(), _Z.data() + (NUM_THREADS - 1) * batch_size * new_nc * new_ht * new_wd,
			batch_size + m % NUM_THREADS, i_ht, i_wd, a_nc, num_filters, _FX, _FY, stride_xy);
	}

#else
	convolve_2d_batch_st(_X.data(), _W.data(), _b.data(), _Z.data(), m, i_ht, i_wd, a_nc, num_filters, _FX, _FY, stride_xy);
#endif
	
	ref_Z = &_Z;
	act_func->Forward_Inp(_Z);

	return true;
}

bool Convolution::Backward_Pass(Tensor4f& dout, Tensor4f& dX)
{
	if (dout.dimension(0) != ref_X->dimension(0))return false;
	const int m = ref_X->dimension(0);
	const int a_nc = ref_X->dimension(1);
	const int a_ht = ref_X->dimension(2);
	const int a_wd = ref_X->dimension(3);
	const int w_nc = _FC;
	const int w_nf = num_filters;
	const int d_nc = num_filters;
	const int d_ht = (a_ht - _FY) / stride_xy + 1;
	const int d_wd = (a_wd - _FX) / stride_xy + 1;
	
	
	if (dout.dimension(1) * dout.dimension(2) * dout.dimension(3) != d_nc * d_ht * d_wd)return false;
	// Reshape dout to appropriate shape //

	act_func->Backprop_Inp(*ref_Z);
	//Tensor4f dZL = *ref_Z * dout;
	dout *= *ref_Z;

	Eigen::Map<Matrix<float, Dynamic, Dynamic, RowMajor>> mat_grads_W(grads_W.data(), w_nf, _FC * _FX * _FY);
	mat_grads_W.setZero();
	grads_b.setZero();
	
	dX.resize(m, a_nc, a_ht, a_wd);
	dX.setZero();

#if defined(EIGEN_DONT_PARALLELIZE)

	Matrix<float, Dynamic, Dynamic, RowMajor > w_grad_acc[NUM_THREADS];
	Matrix<float, Dynamic, Dynamic, RowMajor > b_grad_acc[NUM_THREADS];
	for (int i = 0; i < NUM_THREADS; i++) {
		w_grad_acc[i].resize(w_nf, _FC * _FX * _FY);
		b_grad_acc[i].resize(1, w_nf);
	}

	const int batch_size = m / NUM_THREADS;
	{
		std::future<void> convolve_back[NUM_THREADS];

		for (int i = 0; i < NUM_THREADS - 1; i++)
			convolve_back[i] = std::async(std::launch::async, convolve_back_batch_st, ref_X->data() + i * batch_size * a_nc * a_ht * a_wd,
				dout.data() + i * batch_size * d_nc * d_ht * d_wd, _W.data(), w_grad_acc[i].data(), b_grad_acc[i].data(), dX.data() + i * batch_size * a_nc * a_ht * a_wd,
				batch_size, a_nc, a_ht, a_wd, d_nc, d_ht, d_wd, _FX, _FY, _FC, stride_xy);

		convolve_back[NUM_THREADS - 1] = std::async(std::launch::async, convolve_back_batch_st, ref_X->data() + (NUM_THREADS - 1) * batch_size * a_nc * a_ht * a_wd,
			dout.data() + (NUM_THREADS - 1) * batch_size * d_nc * d_ht * d_wd, _W.data(), w_grad_acc[NUM_THREADS - 1].data(), b_grad_acc[NUM_THREADS - 1].data(), dX.data() + (NUM_THREADS - 1) * batch_size * a_nc * a_ht * a_wd,
			batch_size + m % NUM_THREADS, a_nc, a_ht, a_wd, d_nc, d_ht, d_wd, _FX, _FY, _FC, stride_xy);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		mat_grads_W.noalias() += w_grad_acc[i];
		grads_b.noalias() += b_grad_acc[i];
	}
	mat_grads_W.noalias() = mat_grads_W / m;
	grads_b.noalias() = grads_b / m;
#else
	Matrix<float, Dynamic, Dynamic, RowMajor > w_grad_acc(w_nf, _FC * _FX * _FY);
	Matrix<float, 1, Dynamic, RowMajor> b_grad_acc(1, w_nf);

	convolve_back_batch_st(ref_X->data(), dout.data(), _W.data(), w_grad_acc.data(), b_grad_acc.data(), dX.data(),
		m, a_nc, a_ht, a_wd, d_nc, d_ht, d_wd, _FX, _FY, _FC, stride_xy);

	mat_grads_W.noalias() = w_grad_acc / m;
	grads_b.noalias() = b_grad_acc / m;
#endif

	return true;
}
