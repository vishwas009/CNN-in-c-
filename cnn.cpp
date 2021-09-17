#include "cnn.h"

bool load_img_data(char* path, bgra8* img_buff)
{
	jpeg_decompress_struct img_info;
	jpeg_error_mgr err;
	JSAMPARRAY buffer;

	FILE* pFile = fopen(path, "rb");
	if (!pFile)
		return false;

	img_info.err = jpeg_std_error(&err);

	jpeg_create_decompress(&img_info);
	jpeg_stdio_src(&img_info, pFile);
	jpeg_read_header(&img_info, TRUE);
	jpeg_start_decompress(&img_info);

	buffer = (*img_info.mem->alloc_sarray)
		((j_common_ptr)&img_info, JPOOL_IMAGE, img_info.output_width * img_info.output_components, 1);

	int width = img_info.output_width;
	int j = 0;
	while (img_info.output_scanline < img_info.output_height)
	{
		(void)jpeg_read_scanlines(&img_info, buffer, 1);

		// get the pointer to the row:
		unsigned char* pixel_row = (unsigned char*)(buffer[0]);
		// iterate over the pixels:
		for (int i = 0; i < img_info.output_width; i++)
		{
			int p_idx = j * width + i;
			img_buff[p_idx].r = *pixel_row++;
			img_buff[p_idx].g = *pixel_row++;
			img_buff[p_idx].b = *pixel_row++;
		}
		j++;
	}

	jpeg_finish_decompress(&img_info);
	jpeg_destroy_decompress(&img_info);
	fclose(pFile);
	return true;
}

static void convolve_2d(const float* input_img, const float* kernel, float* output_img, const int& ht, const int& wd, const int& f, const int& stride = 1) {
	int n_wd = (wd - f) / stride + 1;
	int n_ht = (ht - f) / stride + 1;
	__m256 p_res1 = _mm256_setzero_ps();
	__m256 p_res2 = _mm256_setzero_ps();
	__m256 p_res3 = _mm256_setzero_ps();
	__m256 p_res4 = _mm256_setzero_ps();
	__m256 p_res5 = _mm256_setzero_ps();
	__m256 p_res6 = _mm256_setzero_ps();
	__m256 p_res7 = _mm256_setzero_ps();
	__m256 p_res8 = _mm256_setzero_ps();
	__m256 brod;

	for (int i = 0; i < n_ht; i++) {
		int j = 0;
		for (j = 0; j <= n_wd - 64; j += 64) {
			p_res1 = _mm256_load_ps(&output_img[i * n_wd + j]);
			p_res2 = _mm256_load_ps(&output_img[i * n_wd + j + 8]);
			p_res3 = _mm256_load_ps(&output_img[i * n_wd + j + 16]);
			p_res4 = _mm256_load_ps(&output_img[i * n_wd + j + 24]);
			p_res5 = _mm256_load_ps(&output_img[i * n_wd + j + 32]);
			p_res6 = _mm256_load_ps(&output_img[i * n_wd + j + 40]);
			p_res7 = _mm256_load_ps(&output_img[i * n_wd + j + 48]);
			p_res8 = _mm256_load_ps(&output_img[i * n_wd + j + 56]);
			for (int fy = 0; fy < f; fy++)
				for (int fx = 0; fx < f; fx++) {
					brod = _mm256_set1_ps(kernel[fy * f + fx]);
					p_res1 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + fx]), brod, p_res1);
					p_res2 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + 8 + fx]), brod, p_res2);
					p_res3 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + 16 + fx]), brod, p_res3);
					p_res4 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + 24 + fx]), brod, p_res4);
					p_res5 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + 32 + fx]), brod, p_res5);
					p_res6 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + 40 + fx]), brod, p_res6);
					p_res7 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + 48 + fx]), brod, p_res7);
					p_res8 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + 56 + fx]), brod, p_res8);
				}
			_mm256_store_ps(&output_img[i * n_wd + j], p_res1);
			_mm256_store_ps(&output_img[i * n_wd + j + 8], p_res2);
			_mm256_store_ps(&output_img[i * n_wd + j + 16], p_res3);
			_mm256_store_ps(&output_img[i * n_wd + j + 24], p_res4);
			_mm256_store_ps(&output_img[i * n_wd + j + 32], p_res5);
			_mm256_store_ps(&output_img[i * n_wd + j + 40], p_res6);
			_mm256_store_ps(&output_img[i * n_wd + j + 48], p_res7);
			_mm256_store_ps(&output_img[i * n_wd + j + 56], p_res8);
		}

		for (; j <= n_wd - 32; j += 32) {
			p_res1 = _mm256_load_ps(&output_img[i * n_wd + j]);
			p_res2 = _mm256_load_ps(&output_img[i * n_wd + j + 8]);
			p_res3 = _mm256_load_ps(&output_img[i * n_wd + j + 16]);
			p_res4 = _mm256_load_ps(&output_img[i * n_wd + j + 24]);
			for (int fy = 0; fy < f; fy++)
				for (int fx = 0; fx < f; fx++) {
					brod = _mm256_set1_ps(kernel[fy * f + fx]);
					p_res1 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + fx]), brod, p_res1);
					p_res2 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + 8 + fx]), brod, p_res2);
					p_res3 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + 16 + fx]), brod, p_res3);
					p_res4 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + 24 + fx]), brod, p_res4);
				}
			_mm256_store_ps(&output_img[i * n_wd + j], p_res1);
			_mm256_store_ps(&output_img[i * n_wd + j + 8], p_res2);
			_mm256_store_ps(&output_img[i * n_wd + j + 16], p_res3);
			_mm256_store_ps(&output_img[i * n_wd + j + 24], p_res4);
		}

		for (; j <= n_wd - 16; j += 16) {
			p_res1 = _mm256_load_ps(&output_img[i * n_wd + j]);
			p_res2 = _mm256_load_ps(&output_img[i * n_wd + j + 8]);
			for (int fy = 0; fy < f; fy++)
				for (int fx = 0; fx < f; fx++) {
					brod = _mm256_set1_ps(kernel[fy * f + fx]);
					p_res1 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + fx]), brod, p_res1);
					p_res2 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + 8 + fx]), brod, p_res2);
				}
			_mm256_store_ps(&output_img[i * n_wd + j], p_res1);
			_mm256_store_ps(&output_img[i * n_wd + j + 8], p_res2);
		}

		for (; j <= n_wd - 8; j += 8) {
			p_res1 = _mm256_load_ps(&output_img[i * n_wd + j]);
			for (int fy = 0; fy < f; fy++)
				for (int fx = 0; fx < f; fx++)
					p_res1 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + fx]), _mm256_set1_ps(kernel[fy * f + fx]), p_res1);

			_mm256_store_ps(&output_img[i * n_wd + j], p_res1);
		}

		if (j < n_wd) {
			p_res1 = _mm256_setzero_ps();
			for (int rmd = j, pi = 0; rmd < n_wd; rmd++)
				p_res1.m256_f32[pi++] = output_img[i * n_wd + rmd];

			for (int fy = 0; fy < f; fy++)
				for (int fx = 0; fx < f; fx++)
					p_res1 = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i * stride + fy) * wd + j * stride + fx]), _mm256_set1_ps(kernel[fy * f + fx]), p_res1);

			for (int pi = 0; j < n_wd; j++)
				output_img[i * n_wd + j] = p_res1.m256_f32[pi++];
		}


	}
}

static void convolve_backward_2d(const float* input_img, const float* kernel, float* output_img, const int& ht, const int& wd, const int& fx, const int& fy, const int& stride = 1) {
	const int n_wd = (wd - fx) / stride + 1;
	const int n_ht = (ht - fy) / stride + 1;
	__m256 acc = _mm256_setzero_ps();
	float res_acc = 0;

	for (int i = 0; i < n_ht; i++)
		for (int j = 0; j < n_wd; j++) {
			res_acc = 0;
			acc = _mm256_setzero_ps();
			for (int ify = 0; ify < fy; ify++) {
				int jfx = 0;
				for (jfx = 0; jfx <= fx - 64; jfx += 64) {
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx]), _mm256_load_ps(&kernel[ify * fx + jfx]), acc);
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx + 8]), _mm256_load_ps(&kernel[ify * fx + jfx + 8]), acc);
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx + 16]), _mm256_load_ps(&kernel[ify * fx + jfx + 16]), acc);
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx + 24]), _mm256_load_ps(&kernel[ify * fx + jfx + 24]), acc);
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx + 32]), _mm256_load_ps(&kernel[ify * fx + jfx + 32]), acc);
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx + 40]), _mm256_load_ps(&kernel[ify * fx + jfx + 40]), acc);
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx + 48]), _mm256_load_ps(&kernel[ify * fx + jfx + 48]), acc);
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx + 56]), _mm256_load_ps(&kernel[ify * fx + jfx + 56]), acc);
				}

				for (; jfx <= fx - 32; jfx += 32) {
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx]), _mm256_load_ps(&kernel[ify * fx + jfx]), acc);
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx + 8]), _mm256_load_ps(&kernel[ify * fx + jfx + 8]), acc);
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx + 16]), _mm256_load_ps(&kernel[ify * fx + jfx + 16]), acc);
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx + 24]), _mm256_load_ps(&kernel[ify * fx + jfx + 24]), acc);

				}
				for (; jfx <= fx - 16; jfx += 16) {
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx]), _mm256_load_ps(&kernel[ify * fx + jfx]), acc);
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx + 8]), _mm256_load_ps(&kernel[ify * fx + jfx + 8]), acc);
				}

				for (; jfx <= fx - 8; jfx += 8)
					acc = _mm256_fmadd_ps(_mm256_load_ps(&input_img[(i + ify) * wd + j + jfx]), _mm256_load_ps(&kernel[ify * fx + jfx]), acc);


				for (; jfx < fx; jfx++)
					res_acc += input_img[(i + ify) * wd + j + jfx] * kernel[ify * fx + jfx];
			}

			res_acc += acc.m256_f32[0];
			res_acc += acc.m256_f32[1];
			res_acc += acc.m256_f32[2];
			res_acc += acc.m256_f32[3];
			res_acc += acc.m256_f32[4];
			res_acc += acc.m256_f32[5];
			res_acc += acc.m256_f32[6];
			res_acc += acc.m256_f32[7];
			output_img[i * n_wd + j] += res_acc;
		}

}

bool Convolve_Forward(const Matrix<float>& a_prev, const Matrix<float>& w_conv, const Matrix<float>& bias, Matrix<float>& output, const int stride)
{
	const Shape ap_shape = a_prev.shape();
	const Shape w_shape = w_conv.shape();
	const int m = ap_shape.d4;
	const int a_nc = ap_shape.d3;
	const int i_ht = ap_shape.d2;
	const int i_wd = ap_shape.d1;
	const int f = w_shape.d1;
	const int w_nc = w_shape.d3;
	const int new_nc = w_shape.d4;
	const int new_wd = (i_wd - f) / stride + 1;
	const int new_ht = (i_ht - f) / stride + 1;
	const int row_len = new_wd * new_ht;
	if (a_nc != w_nc) return false;
	
	output.resize({ m,new_nc,new_ht,new_wd });
	const float* ap_ptr = a_prev.data();
	const float* wc_ptr = w_conv.data();
	const float* b_ptr = bias.data();
	float* out_ptr = output.data();

#pragma omp parallel for
	for (int n = 0; n < m; n++) {
		int ap_ofst = n * a_nc * i_wd * i_ht;
		int out_ofst = n * new_nc * row_len;
		for (int z = 0; z < new_nc; z++) {
			memset(&out_ptr[out_ofst + z * row_len], 0, row_len * sizeof(float));
			for (int n_c = 0; n_c < a_nc; n_c++)
				convolve_2d(&ap_ptr[ap_ofst + n_c * i_wd * i_ht], &wc_ptr[z * w_nc * f * f + n_c * f * f], &out_ptr[out_ofst + z * row_len], i_ht, i_wd, f, stride);

			__m256 brod = _mm256_set1_ps(b_ptr[z]);
			int len = 0;
			for (len = 0; len <= row_len - 16; len += 16) {
				_mm256_store_ps(&out_ptr[out_ofst + z * row_len + len], _mm256_add_ps(_mm256_load_ps(&out_ptr[out_ofst + z * row_len + len]), brod));
				_mm256_store_ps(&out_ptr[out_ofst + z * row_len + len + 8], _mm256_add_ps(_mm256_load_ps(&out_ptr[out_ofst + z * row_len + len + 8]), brod));
			}
			for (; len <= row_len - 8; len += 8)
				_mm256_store_ps(&out_ptr[out_ofst + z * row_len + len], _mm256_add_ps(_mm256_load_ps(&out_ptr[out_ofst + z * row_len + len]), brod));

			for (; len < row_len; len++)
				out_ptr[out_ofst + z * row_len + len] += b_ptr[z];
		}
	}

	return true;
}

bool Convolve_Forward_v2(const Matrix<float>& a_prev, const Matrix<float>& w_conv, const Matrix<float>& bias, Matrix<float>& output, const int stride)
{
	
	const Shape ap_shape = a_prev.shape();
	const Shape w_shape = w_conv.shape();
	const int m = ap_shape.d4;
	const int a_nc = ap_shape.d3;
	const int i_ht = ap_shape.d2;
	const int i_wd = ap_shape.d1;
	const int f = w_shape.d1;
	const int w_nc = w_shape.d3;
	const int new_nc = w_shape.d4;
	const int new_wd = (i_wd - f) / stride + 1;
	const int new_ht = (i_ht - f) / stride + 1;
	const int row_len = new_wd * new_ht;
	if (a_nc != w_nc) return false;

	output.resize({ m,new_nc,new_ht,new_wd });
	const float* ap_ptr = a_prev.data();
	const float* wc_ptr = w_conv.data();
	const float* b_ptr = bias.data();
	float* out_ptr = output.data();
	float* a_col_slc = (float*)mkl_malloc(a_nc * f * f * NUM_THREADS * new_ht * new_wd * sizeof(float), 64);

	auto convolve_slc = [=](float* a_col, int batch, int offset) {
		for (int n = offset; n < offset + batch; n++) {
			const int ap_exm_ofst = n * a_nc * i_ht * i_wd;
			const int ac_chn_ofst = new_ht * new_wd;
			const int num_el = sizeof(float) * new_wd;
			for (int y = 0; y < new_ht; y++)
				for (int z = 0; z < a_nc; z++)
					for (int fy = 0; fy < f; fy++)
						for (int fx = 0; fx < f; fx++)
							memcpy(&a_col[(z * f * f + fy * f + fx) * ac_chn_ofst + y * new_wd], &ap_ptr[ap_exm_ofst + z * i_ht * i_wd + (y + fy) * i_wd + fx], num_el);

			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				new_nc, new_ht * new_wd, a_nc * f * f,
				1.0, wc_ptr, a_nc * f * f, a_col, new_ht * new_wd,
				0.0, out_ptr + n * new_nc * new_ht * new_wd, new_ht * new_wd);

			const int row_len = new_ht * new_wd;
			const int az_exm_ofst = n * new_nc * new_ht * new_wd;
			for (int z = 0; z < new_nc; z++) {
				__m256 bias = _mm256_set1_ps(b_ptr[z]);
				int i = 0;
				for (i = 0; i <= row_len - 8; i += 8)
					_mm256_store_ps(&out_ptr[az_exm_ofst + z * row_len + i], _mm256_add_ps(_mm256_load_ps(&out_ptr[az_exm_ofst + z * row_len + i]), bias));

				for (int rmd = i; rmd < row_len; rmd++)
					out_ptr[az_exm_ofst + z * row_len + rmd] += b_ptr[z];
			}

		}
	};

	{
		int batch_size = m / NUM_THREADS;
		std::future<void> convolve_calc[NUM_THREADS];

		for (int i = 0; i < NUM_THREADS - 1; i++)
			convolve_calc[i] = std::async(std::launch::async, convolve_slc, a_col_slc + i * a_nc * f * f * new_ht * new_wd, batch_size, i * batch_size);

		convolve_calc[NUM_THREADS - 1] = std::async(std::launch::async, convolve_slc, a_col_slc + (NUM_THREADS - 1) * a_nc * f * f * new_ht * new_wd, batch_size + m % NUM_THREADS, (NUM_THREADS - 1) * batch_size);
	}

	mkl_free(a_col_slc);
	return true;

}

bool Convolve_Backward(const Matrix<float>& aX, const Matrix<float>& dout, const Matrix<float>& w_conv, Matrix<float>& w_grad, Matrix<float>& b_grad, Matrix<float>& dX, const int stride)
{
	const Shape aX_shape = aX.shape();
	const Shape dout_shape = dout.shape();
	const Shape w_shape = w_conv.shape();
	const int m = aX_shape.d4;
	const int a_nc = aX_shape.d3;
	const int a_ht = aX_shape.d2;
	const int a_wd = aX_shape.d1;
	const int f = w_shape.d1;
	const int w_nc = w_shape.d3;
	const int w_nf = w_shape.d4;
	const int d_nc = dout_shape.d3;
	const int d_ht = dout_shape.d2;
	const int d_wd = dout_shape.d1;
	if (a_nc != w_nc) { std::cerr << "\nError during convolve backward\n"; return false; }
	if (d_nc != w_nf) { std::cerr << "\nError during convolve backward\n"; return false; }
	const int f_wd = (a_wd - d_wd) / stride + 1;
	const int f_ht = (a_ht - d_ht) / stride + 1;
	if (f != f_wd || f != f_ht) { std::cerr << "\nError during convolve backward\n"; return false; }

	// Turn aX into aX_col Im2col part //
	//auto start = std::chrono::high_resolution_clock::now();
	float* aX_col = (float*)mkl_malloc(a_nc * f * f * m * d_ht * d_wd * sizeof(float), 64);
	const float* aX_ptr = aX.data();
	
#pragma omp parallel for
	for (int n = 0; n < m; n++) {
		const int ap_exm_ofst = n * a_nc * a_ht * a_wd;
		const int ac_chn_ofst = n * d_ht * d_wd;
		const int ac_exm_ofst = a_nc * f * f;
		for (int y = 0; y < d_ht; y++)
			for (int x = 0; x < d_wd; x++)
				for (int z = 0; z < a_nc; z++)
					for (int fy = 0; fy < f; fy++)
						for (int fx = 0; fx < f; fx++)
							aX_col[(ac_chn_ofst + y * d_wd + x) * ac_exm_ofst + z * f * f + fy * f + fx] = aX_ptr[ap_exm_ofst + z * a_ht * a_wd + (y + fy) * a_wd + fx + x];

	}
	//auto stop = std::chrono::high_resolution_clock::now();
	//auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	// Turning dout into dout_col //
	//start = std::chrono::high_resolution_clock::now();
	float* dout_col = (float*)mkl_malloc(d_nc * d_ht * d_wd * m * sizeof(float), 64);
	const float* dout_ptr = dout.data();

#pragma omp parallel for
	for (int n = 0; n < m; n++)
		for (int z = 0; z < d_nc; z++)
			memcpy(&dout_col[z * m * d_ht * d_wd + n * d_ht * d_wd], &dout_ptr[n * d_nc * d_ht * d_wd + z * d_ht * d_wd], sizeof(float) * d_ht * d_wd);

	//stop = std::chrono::high_resolution_clock::now();
	//auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	// Calculating w_grads by Matrix Multiplication between dout_col and aX_col //
	//start = std::chrono::high_resolution_clock::now();
	float* w_grad_ptr = w_grad.data();
	const int dm = d_nc;
	const int dn = a_nc * f * f;
	const int dk = m * d_ht * d_wd;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		dm, dn, dk, 1.0, dout_col,
		dk, aX_col,
		dn, 0.0, w_grad_ptr, dn);

	mkl_free(aX_col);
	//stop = std::chrono::high_resolution_clock::now();
	//auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	// Calcuting b_grads //
	//start = std::chrono::high_resolution_clock::now();
	b_grad.set_Zero();
	float* b_grad_ptr = b_grad.data();
	const int row_len = m * d_ht * d_wd;
	__m256 p_sum = _mm256_setzero_ps();
	for (int z = 0; z < d_nc; z++) {
		int i = 0;
		p_sum = _mm256_load_ps(&dout_col[z * row_len]);
		for (i = 8; i <= row_len - 8; i += 8)
			p_sum = _mm256_add_ps(_mm256_load_ps(&dout_col[z * row_len + i]), p_sum);

		for (; i < row_len; i++)
			b_grad_ptr[z] += dout_col[z * row_len + i];

		for (int p = 0; p < 8; p++)
			b_grad_ptr[z] += p_sum.m256_f32[p];
	}
	//stop = std::chrono::high_resolution_clock::now();
	//auto t4 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	// Scale Gradinets calculated by number of Examples //
	for (int i = 0; i < dn * dm; i++)
		w_grad_ptr[i] /= m;

	for (int i = 0; i < d_nc; i++)
		b_grad_ptr[i] /= m;

	// Calculating dX by Matrix Multiplication between dout_col and W_Transpose //
	//start = std::chrono::high_resolution_clock::now();
	const float* w_conv_ptr = w_conv.data();
	float* dX_col = (float*)mkl_malloc(dn * dk * sizeof(float), 64);
	float* w_col_T = (float*)mkl_malloc(dn * dm * sizeof(float), 64);
	mkl_somatcopy('R', 'T', dm, dn, 1.0, w_conv_ptr, dn, w_col_T, dm); // Transposing W //

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		a_nc * f * f, m * d_ht * d_wd, d_nc, 1.0, w_col_T,
		d_nc, dout_col,
		m * d_ht * d_wd, 0.0, dX_col, m * d_ht * d_wd);

	mkl_free(w_col_T);
	//stop = std::chrono::high_resolution_clock::now();
	//auto t5 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	// Turning dX_col into correct shape by Col2im method //
	//start = std::chrono::high_resolution_clock::now();
	dX.resize(aX_shape);
	dX.set_Zero();
	float* dX_ptr = dX.data();
	int dx_chn_ofst = 0;
	int dxc_chn_ofst = 0;
	int dxc_exm_ofst = m * d_ht * d_wd;
	int dx_exm_ofst = 0;
	int dxc_img_ofst = 0;
	for (int z = 0; z < a_nc; z++) {
		dx_chn_ofst = z * a_ht * a_wd;
		dxc_chn_ofst = z * f * f;
		for (int fy = 0; fy < f; fy++) {
			for (int fx = 0; fx < f; fx++) {
				for (int n = 0; n < m; n++) {
					dxc_img_ofst = n * d_ht * d_wd;
					dx_exm_ofst = n * a_nc * a_ht * a_wd;
					for (int y = 0; y < d_ht; y++) {
						int x = 0;
						for (x = 0; x <= d_wd - 8; x += 8)
							_mm256_store_ps(&dX_ptr[dx_exm_ofst + dx_chn_ofst + (y + fy) * a_wd + x + fx],
								_mm256_add_ps(_mm256_load_ps(&dX_ptr[dx_exm_ofst + dx_chn_ofst + (y + fy) * a_wd + x + fx]),
									_mm256_load_ps(&dX_col[(dxc_chn_ofst + fy * f + fx) * dxc_exm_ofst + dxc_img_ofst + y * d_wd + x])));

						for (; x < d_wd; x++)
							dX_ptr[dx_exm_ofst + dx_chn_ofst + (y + fy) * a_wd + x + fx] += dX_col[(dxc_chn_ofst + fy * f + fx) * dxc_exm_ofst + dxc_img_ofst + y * d_wd + x];

					}
				}
			}
		}
	}
	//auto stop = std::chrono::high_resolution_clock::now();
	//auto t6 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	mkl_free(dX_col);
	mkl_free(dout_col);
	//std::cout << "\nTime 1:  " << t6.count() << "\n\n";
	//std::cout << "\nTime 1:  " << t1.count() << "  Time 2:  " << t2.count() << "  Time 3:  " << t3.count() << "  Time 4:  " << t4.count() << "  Time 5:  " << t5.count() << "  Time 6:  " << t6.count() << "\n\n";
	return true;
}

bool Convolve_Backward_v2(const Matrix<float>& aX, const Matrix<float>& dout, const Matrix<float>& w_conv, Matrix<float>& w_grad, Matrix<float>& b_grad, Matrix<float>& dX, const int stride)
{
	const Shape aX_shape = aX.shape();
	const Shape dout_shape = dout.shape();
	const Shape w_shape = w_conv.shape();
	const int m = aX_shape.d4;
	const int a_nc = aX_shape.d3;
	const int a_ht = aX_shape.d2;
	const int a_wd = aX_shape.d1;
	const int f = w_shape.d1;
	const int w_nc = w_shape.d3;
	const int w_nf = w_shape.d4;
	const int d_nc = dout_shape.d3;
	const int d_ht = dout_shape.d2;
	const int d_wd = dout_shape.d1;
	if (a_nc != w_nc) { std::cerr << "\nError during convolve backward\n"; return false; }
	if (d_nc != w_nf) { std::cerr << "\nError during convolve backward\n"; return false; }
	const int f_wd = (a_wd - d_wd) / stride + 1;
	const int f_ht = (a_ht - d_ht) / stride + 1;
	if (f != f_wd || f != f_ht) { std::cerr << "\nError during convolve backward\n"; return false; }

	int pad_x = ((a_wd - 1) * stride - d_wd + f) / 2;
	int pad_y = ((a_ht - 1) * stride - d_ht + f) / 2;

	float* w_rot_180 = (float*)mkl_malloc(d_nc * a_nc * f * f * sizeof(float), 64);
	memset(w_rot_180, 0, d_nc * a_nc * f * f * sizeof(float));
	const float* w_conv_ptr = w_conv.data();

	for (int n = 0; n < d_nc; n++)
		for (int z = 0; z < a_nc; z++)
			for (int fy = 0; fy < f; fy++)
				for (int fx = 0; fx < f; fx++)
					w_rot_180[n * a_nc * f * f + z * f * f + (f - 1 - fy) * f + f - 1 - fx] = w_conv_ptr[n * a_nc * f * f + z * f * f + fy * f + fx];

	auto start = std::chrono::high_resolution_clock::now();
	const int row_len = d_ht * d_wd;
	const float* aX_ptr = aX.data();
	const float* dout_ptr = dout.data();
	float* w_grad_ptr = w_grad.data();
	float* b_grad_ptr = b_grad.data();
	w_grad.set_Zero();

#pragma omp parallel for
	for (int z = 0; z < d_nc; z++) {
		b_grad_ptr[z] = 0;
		for (int n = 0; n < m; n++) {
			for (int nc = 0; nc < a_nc; nc++)
				convolve_backward_2d(&aX_ptr[n * a_nc * a_ht * a_wd + nc * a_ht * a_wd],
					&dout_ptr[n * d_nc * d_ht * d_wd + z * d_ht * d_wd], &w_grad_ptr[z * a_nc * f * f + nc * f * f], a_ht, a_wd, d_wd, d_ht, stride);
			__m256 b_sum = _mm256_setzero_ps();
			int i = 0;
			float acc = 0;
			for (i = 0; i <= row_len - 8; i += 8)
				b_sum = _mm256_add_ps(_mm256_load_ps(&dout_ptr[n * d_nc * d_ht * d_wd + z * d_ht * d_wd + i]), b_sum);

			for (; i < row_len; i++)
				acc += dout_ptr[n * d_nc * d_ht * d_wd + z * d_ht * d_wd + i];

			acc += b_sum.m256_f32[0];
			acc += b_sum.m256_f32[1];
			acc += b_sum.m256_f32[2];
			acc += b_sum.m256_f32[3];
			acc += b_sum.m256_f32[4];
			acc += b_sum.m256_f32[5];
			acc += b_sum.m256_f32[6];
			acc += b_sum.m256_f32[7];
			b_grad_ptr[z] += acc;
		}
	}
	// Scale Gradinets calculated by number of Examples //
	const int wg_size = w_grad.size();
	for (int i = 0; i < wg_size; i++)
		w_grad_ptr[i] /= m;

	for (int i = 0; i < d_nc; i++)
		b_grad_ptr[i] /= m;

	float* dout_pad_slc = (float*)mkl_malloc(NUM_THREADS * (d_ht + pad_y * 2) * (d_wd + pad_x * 2) * sizeof(float), 64);

	auto dX_calc = [=](float* dout_pad, float* w_180, float* dX, int batch, int offset) {
		for (int n = offset; n < offset + batch; n++) {
			for (int z = 0; z < d_nc; z++) {
				for (int p1 = 0; p1 < pad_y; p1++)
					memset(&dout_pad[p1 * (d_wd + pad_x * 2)], 0, sizeof(float) * (d_wd + pad_x * 2));

				for (int y = 0; y < d_ht; y++) {
					for (int p1 = 0; p1 < pad_x; p1++)
						dout_pad[(y + pad_y) * (d_wd + pad_x * 2) + p1] = 0;

					memcpy(&dout_pad[(y + pad_y) * (d_wd + pad_x * 2) + pad_x],
						&dout_ptr[n * d_nc * d_ht * d_wd + z * d_ht * d_wd + y * d_wd], d_wd * sizeof(float));

					for (int p2 = 0; p2 < pad_x; p2++)
						dout_pad[(y + pad_y) * (d_wd + pad_x * 2) + pad_x + d_wd + p2] = 0;
				}

				for (int p2 = 0; p2 < pad_y; p2++)
					memset(&dout_pad[(pad_y + d_ht + p2) * (d_wd + pad_x * 2)], 0, sizeof(float) * (d_wd + pad_x * 2));

				for (int nc = 0; nc < a_nc; nc++)
					convolve_2d(dout_pad, &w_rot_180[z * a_nc * f * f + nc * f * f],
						&dX[n * a_nc * a_ht * a_wd + nc * a_ht * a_wd], (d_ht + pad_y * 2), (d_wd + pad_x * 2), f, stride);
			}

		}
	};

	dX.resize(aX_shape);
	dX.set_Zero();
	float* dX_ptr = dX.data();

	{
		int batch_size = m / NUM_THREADS;
		std::future<void> dx_calc[NUM_THREADS];

		for (int i = 0; i < NUM_THREADS - 1; i++)
			dx_calc[i] = std::async(std::launch::async, dX_calc, dout_pad_slc + i * (d_ht + pad_y * 2) * (d_wd + pad_x * 2), w_rot_180, dX_ptr, batch_size, i * 32);

		dx_calc[NUM_THREADS - 1] = std::async(std::launch::async, dX_calc, dout_pad_slc + (NUM_THREADS - 1) * (d_ht + pad_y * 2) * (d_wd + pad_x * 2), w_rot_180, dX_ptr, batch_size + m % NUM_THREADS, (NUM_THREADS - 1) * 32);

	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	std::cout << "\nTime :  " << t1.count() << "\n\n";

	mkl_free(dout_pad_slc);
	mkl_free(w_rot_180);
	return true;
}

bool Convolve_Backward_v3(const Matrix<float>& aX, const Matrix<float>& dout, const Matrix<float>& w_conv, Matrix<float>& w_grad, Matrix<float>& b_grad, Matrix<float>& dX, const int stride)
{
	const Shape aX_shape = aX.shape();
	const Shape dout_shape = dout.shape();
	const Shape w_shape = w_conv.shape();
	const int m = aX_shape.d4;
	const int a_nc = aX_shape.d3;
	const int a_ht = aX_shape.d2;
	const int a_wd = aX_shape.d1;
	const int f = w_shape.d1;
	const int w_nc = w_shape.d3;
	const int w_nf = w_shape.d4;
	const int d_nc = dout_shape.d3;
	const int d_ht = dout_shape.d2;
	const int d_wd = dout_shape.d1;
	if (a_nc != w_nc) { std::cerr << "\nError during convolve backward\n"; return false; }
	if (d_nc != w_nf) { std::cerr << "\nError during convolve backward\n"; return false; }
	const int f_wd = (a_wd - d_wd) / stride + 1;
	const int f_ht = (a_ht - d_ht) / stride + 1;
	if (f != f_wd || f != f_ht) { std::cerr << "\nError during convolve backward\n"; return false; }

	const float* w_conv_ptr = w_conv.data();
	const float* aX_ptr = aX.data();
	const float* dout_ptr = dout.data();
	float* aX_col_slc = (float*)mkl_malloc(a_nc * f * f * NUM_THREADS * d_ht * d_wd * sizeof(float), 64);
	float* dX_col_slc = (float*)mkl_malloc(a_nc * f * f * NUM_THREADS * d_ht * d_wd * sizeof(float), 64);
	float* w_grad_acc_slc = (float*)mkl_malloc(NUM_THREADS * a_nc * f * f * d_nc * sizeof(float), 64);
	float* b_grad_acc_slc = (float*)mkl_malloc(NUM_THREADS * sizeof(float) * d_nc, 64);
	float* w_col_T = (float*)mkl_malloc(a_nc * f * f * d_nc * sizeof(float), 64);
	mkl_somatcopy('R', 'T', d_nc, a_nc * f * f, 1.0, w_conv_ptr, a_nc * f * f, w_col_T, d_nc);
	dX.resize(aX_shape);
	dX.set_Zero();
	float* dX_ptr = dX.data();

	auto convolve_back_slc = [=](float* aX_col, float* dX_col, float* w_grad_acc, float* b_grad_acc, int batch, int offset) {
		memset(dX_col, 0, a_nc * f * f * d_ht * d_wd * sizeof(float));
		memset(w_grad_acc, 0, a_nc * f * f * d_nc * sizeof(float));
		memset(b_grad_acc, 0, sizeof(float) * d_nc);

		for (int n = offset; n < offset + batch; n++) {
			const int ap_exm_ofst = n * a_nc * a_ht * a_wd;
			const int ac_exm_ofst = a_nc * f * f;
			for (int y = 0; y < d_ht; y++)
				for (int x = 0; x < d_wd; x++)
					for (int z = 0; z < a_nc; z++)
						for (int fy = 0; fy < f; fy++)
							for (int fx = 0; fx < f; fx++)
								aX_col[(y * d_wd + x) * ac_exm_ofst + z * f * f + fy * f + fx] = aX_ptr[ap_exm_ofst + z * a_ht * a_wd + (y + fy) * a_wd + fx + x];

			const int dm = d_nc;
			const int dn = a_nc * f * f;
			const int dk = d_ht * d_wd;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				dm, dn, dk, 1.0, dout_ptr + n * d_nc * d_ht * d_wd,
				dk, aX_col,
				dn, 1.0, w_grad_acc, dn);

			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				a_nc * f * f, d_ht * d_wd, d_nc, 1.0, w_col_T,
				d_nc, dout_ptr + n * d_nc * d_ht * d_wd,
				d_ht * d_wd, 0.0, dX_col, d_ht * d_wd);

			int dx_chn_ofst = 0;
			int dxc_chn_ofst = 0;
			int dx_exm_ofst = 0;
			int dxc_img_ofst = 0;
			for (int z = 0; z < a_nc; z++) {
				dx_chn_ofst = z * a_ht * a_wd;
				dxc_chn_ofst = z * f * f;
				for (int fy = 0; fy < f; fy++) {
					for (int fx = 0; fx < f; fx++) {
						dxc_img_ofst = d_ht * d_wd;
						dx_exm_ofst = n * a_nc * a_ht * a_wd;
						for (int y = 0; y < d_ht; y++) {
							int x = 0;
							for (x = 0; x <= d_wd - 8; x += 8)
								_mm256_store_ps(&dX_ptr[dx_exm_ofst + dx_chn_ofst + (y + fy) * a_wd + x + fx],
									_mm256_add_ps(_mm256_load_ps(&dX_ptr[dx_exm_ofst + dx_chn_ofst + (y + fy) * a_wd + x + fx]),
										_mm256_load_ps(&dX_col[(dxc_chn_ofst + fy * f + fx) * dxc_img_ofst + y * d_wd + x])));

							for (; x < d_wd; x++)
								dX_ptr[dx_exm_ofst + dx_chn_ofst + (y + fy) * a_wd + x + fx] += dX_col[(dxc_chn_ofst + fy * f + fx) * dxc_img_ofst + y * d_wd + x];

						}
					}
				}
			}

			const int row_len = d_ht * d_wd;
			__m256 p_sum = _mm256_setzero_ps();
			for (int z = 0; z < d_nc; z++) {
				int i = 0;
				p_sum = _mm256_load_ps(&dout_ptr[n * d_nc * row_len + z * row_len]);
				for (i = 8; i <= row_len - 8; i += 8)
					p_sum = _mm256_add_ps(_mm256_load_ps(&dout_ptr[n * d_nc * row_len + z * row_len + i]), p_sum);

				for (; i < row_len; i++)
					b_grad_acc[z] += dout_ptr[n * d_nc * row_len + z * row_len + i];

				for (int p = 0; p < 8; p++)
					b_grad_acc[z] += p_sum.m256_f32[p];
			}

		}
	};

	{
		int batch_size = m / NUM_THREADS;
		std::future<void> convolve_back[NUM_THREADS];

		for (int i = 0; i < NUM_THREADS - 1; i++)
			convolve_back[i] = std::async(std::launch::async, convolve_back_slc, aX_col_slc + i * a_nc * f * f * d_ht * d_wd,
				dX_col_slc + i * a_nc * f * f * d_ht * d_wd, w_grad_acc_slc + i * a_nc * f * f * d_nc, b_grad_acc_slc + i * d_nc, batch_size, i * batch_size);

		convolve_back[NUM_THREADS - 1] = std::async(std::launch::async, convolve_back_slc, aX_col_slc + (NUM_THREADS - 1) * a_nc * f * f * d_ht * d_wd,
			dX_col_slc + (NUM_THREADS - 1) * a_nc * f * f * d_ht * d_wd, w_grad_acc_slc + (NUM_THREADS - 1) * a_nc * f * f * d_nc,
			b_grad_acc_slc + (NUM_THREADS - 1) * d_nc, batch_size + m % NUM_THREADS, (NUM_THREADS - 1) * batch_size);
	}

	w_grad.set_Zero();
	b_grad.set_Zero();
	float* w_grad_ptr = w_grad.data();
	float* b_grad_ptr = b_grad.data();
	for (int n = 0; n < NUM_THREADS; n++) {
		const int len = a_nc * f * f * d_nc;
		for (int i = 0; i < len; i++)
			w_grad_ptr[i] += w_grad_acc_slc[n * len + i];

		for (int j = 0; j < d_nc; j++)
			b_grad_ptr[j] += b_grad_acc_slc[n * d_nc + j];
	}

	int w_size = w_grad.size();
	for (int i = 0; i < w_size; i++)
		w_grad_ptr[i] /= m;

	for (int i = 0; i < d_nc; i++)
		b_grad_ptr[i] /= m;

	mkl_free(aX_col_slc);
	mkl_free(dX_col_slc);
	mkl_free(w_col_T);
	mkl_free(w_grad_acc_slc);
	mkl_free(b_grad_acc_slc);

	return true;
}

Matrix<float> Max_Pool_Forward(const Matrix<float>& a_prev, const int f, const int stride)
{
	const Shape ap_shape = a_prev.shape();
	const int m = ap_shape.d4;
	const int a_nc = ap_shape.d3;
	const int i_ht = ap_shape.d2;
	const int i_wd = ap_shape.d1;
	const int new_wd = (i_wd - f) / stride + 1;
	const int new_ht = (i_ht - f) / stride + 1;

	Matrix<float> output({ m,a_nc,new_ht,new_wd });
	float t_max = 0;
	const float* ap_ptr = a_prev.data();
	float* out_ptr = output.data();

	for (int n = 0; n < m; n++) {
		//float t_max = 0;
		int ap_exm_ofst = n * a_nc * i_ht * i_wd;
		int out_exm_ofst = n * a_nc * new_ht * new_wd;
		for (int z = 0; z < a_nc; z++) {
			int ap_ofst = ap_exm_ofst + z * i_ht * i_wd;
			int out_ofst = out_exm_ofst + z * new_ht * new_wd;
			for (int y = 0; y < new_ht; y++)
				for (int x = 0; x < new_wd; x++) {
					t_max = ap_ptr[ap_ofst + (y * stride) * i_wd + x * stride];
					for (int fy = 0; fy < f; fy++)
						for (int fx = 0; fx < f; fx++)
							if (t_max < ap_ptr[ap_ofst + (y * stride + fy) * i_wd + x * stride + fx])
								t_max = ap_ptr[ap_ofst + (y * stride + fy) * i_wd + x * stride + fx];
					out_ptr[out_ofst + y * new_wd + x] = t_max;
				}

		}
	}
	
	return output;
}

Matrix<float> Max_Pool_Backward(const Matrix<float>& aX, Matrix<float>& dout, const int f, const int stride)
{
	const Shape aX_shape = aX.shape();
	const Shape dout_shape = dout.shape();
	const int m = aX_shape.d4;
	const int a_nc = aX_shape.d3;
	const int a_ht = aX_shape.d2;
	const int a_wd = aX_shape.d1;
	const int d_ht = dout_shape.d2;
	const int d_wd = dout_shape.d1;

	Matrix<float> dX(aX_shape);
	dX.set_Zero();

	const float* aX_ptr = aX.data();
	const float* dout_ptr = dout.data();
	float* dX_ptr = dX.data();

	for (int n = 0; n < m; n++) {
		float t_max = 0;
		int pos = 0;
		int aX_exm_ofst = n * a_nc * a_ht * a_wd;
		int dout_exm_ofst = n * a_nc * d_ht * d_wd;
		for (int z = 0; z < a_nc; z++) {
			int aX_ofst = aX_exm_ofst + z * a_ht * a_wd;
			int dout_ofst = dout_exm_ofst + z * d_ht * d_wd;
			for (int y = 0; y < d_ht; y++)
				for (int x = 0; x < d_wd; x++) {
					pos = aX_ofst + (y * stride) * a_wd + x * stride;
					t_max = aX_ptr[pos];
					for (int fy = 0; fy < f; fy++)
						for (int fx = 0; fx < f; fx++)
							if (t_max < aX_ptr[aX_ofst + (y * stride + fy) * a_wd + x * stride + fx]) {
								pos = aX_ofst + (y * stride + fy) * a_wd + x * stride + fx;
								t_max = aX_ptr[pos];
							}
					dX_ptr[pos] += dout_ptr[dout_ofst + y * d_wd + x];

				}
		}
	}

	return dX;
}

Matrix<float> Relu(const Matrix<float>& z)
{
	Matrix<float> output(z.shape());
	int n = z.size();
	const float* ap_ptr = z.data();
	float* out_ptr = output.data();

#pragma omp parallel for
	for (int i = 0; i < n; i++)
		out_ptr[i] = (ap_ptr[i] > 0) ? ap_ptr[i] : 0;

	return output;
}

void Relu_Inp(Matrix<float>& z)
{
	int n = z.size();
	float* z_ptr = z.data();

#pragma omp parallel for
	for (int i = 0; i < n; i++)
		z_ptr[i] = (z_ptr[i] > 0) ? z_ptr[i] : 0;

}

Matrix<float> Relu_Gradient(const Matrix<float>& z)
{
	Matrix<float> output(z.shape());
	int n = z.size();
	const float* z_ptr = z.data();
	float* out_ptr = output.data();

#pragma omp parallel for
	for (int i = 0; i < n; i++)
		out_ptr[i] = (z_ptr[i] > 0);

	return output;
}

void Relu_Gradient_v2(Matrix<float>& z)
{
	int n = z.size();
	float* z_ptr = z.data();

#pragma omp parallel for
	for (int i = 0; i < n; i++)
		z_ptr[i] = (z_ptr[i] != 0);

}

Matrix<float> Sigmoid(const Matrix<float>& z)
{
	Matrix output = -z;
	output = output.exp();
	int n = output.size();
	float* out_ptr = output.data();
	for (int i = 0; i < n; i++)
		out_ptr[i] = 1.0 / (1.0 + out_ptr[i]);

	return output;
}

void Sigmoid_Inp(Matrix<float>& z)
{
	int n = z.size();
	float* z_ptr = z.data();
	for (int i = 0; i < n; i++)
		z_ptr[i] = 1.0 / (1.0 + expf(-z_ptr[i]));

}

Matrix<float> Sigmoid_Gradient(const Matrix<float>& z)
{
	Matrix output = -z;
	output = output.exp();
	int n = output.size();
	float* out_ptr = output.data();
	float tmp = 0;
	for (int i = 0; i < n; i++) {
		tmp = 1.0 / (1.0 + out_ptr[i]);
		out_ptr[i] = tmp * (1.0 - tmp);
	}

	return output;
}

void Sigmoid_Gradient_v2(Matrix<float>& z)
{
	int n = z.size();
	float* z_ptr = z.data();
	for (int i = 0; i < n; i++)
		z_ptr[i] = z_ptr[i] * (1.0 - z_ptr[i]);

}

float Cross_Entropy_Cost(Matrix<float>& hx, Matrix<float>& Y)
{
	const int m = hx.shape().d2;
	Matrix term1 = Y * hx.log_e();
	Matrix<float> term2(term1.shape());
	for (int i = 0; i < m; i++)
		term2(i, 0) = (1.0 - Y(i, 0)) * logf(1.0 - hx(i, 0));

	float cost = -(term1 + term2).sum();
	cost /= m;

	return cost;
}

void He_Initialization(Parameters<float>& theta) /// Experimental Demo Implimentation ////
{
	for (int i = 0; i < 2; i++) {
		theta.W[i].set_Random();
		theta.W[i] = theta.W[i] * (float)0.07;
	}
	theta.W[2].set_Random();
	theta.W[2] = theta.W[2] * sqrtf(1.0 / 28800.0);
	theta.W[3].set_Random();
	theta.W[3] = theta.W[3] * sqrtf(1.0 / 64.0);
}

Matrix<float> Forward_Pass(const Matrix<float>& X, const Parameters<float>& weights)
{
	Matrix<float> z1;
	if (!Convolve_Forward(X, weights.W[0], weights.b[0], z1, 1)) {
		std::cerr << "\nError During Convolution\n";
		return Matrix<float>();
	}
	
	Matrix a1 = Relu(z1);
	Matrix a1_pool = Max_Pool_Forward(a1, 2, 2);

	Matrix<float> z2;
	if (!Convolve_Forward(a1_pool, weights.W[1], weights.b[1], z2, 1)) {
		std::cerr << "\nError During Convolution\n";
		return Matrix<float>();
	}
	
	Matrix a2 = Relu(z2);
	Matrix a2_pool = Max_Pool_Forward(a2, 2, 2);

	Shape a2p_shape = a2_pool.shape();
	a2_pool.reshape({ a2p_shape.d4, a2p_shape.d3 * a2p_shape.d2 * a2p_shape.d1 });

	Matrix z3 = a2_pool % weights.W[2];
	int cols = z3.shape().d1;
	int rows = z3.shape().d2;
	float* z3_ptr = z3.data();
	const float* b_ptr = weights.b[2].data();

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			z3_ptr[i * cols + j] += b_ptr[j];

	Matrix a3 = Relu(z3);

	Matrix hx = Sigmoid(a3 % weights.W[3] + weights.b[3](0, 0));

	return hx;
}

Matrix<float> Backward_Pass(const Matrix<float>& X, Matrix<float>& Y, const Parameters<float>& weights, Parameters<float>& w_grads, const float lambda)
{
	// Forward Pass //
	Matrix<float> a1;
	if (!Convolve_Forward_v2(X, weights.W[0], weights.b[0], a1, 1)) {
		std::cerr << "\nError During Convolution\n";
		return Matrix<float>();
	}
	
	Relu_Inp(a1);
	Matrix a1_pool = Max_Pool_Forward(a1, 2, 2);

	Matrix<float> a2;
	if (!Convolve_Forward_v2(a1_pool, weights.W[1], weights.b[1], a2, 1)) {
		std::cerr << "\nError During Convolution\n";
		return Matrix<float>();
	}

	Relu_Inp(a2);
	Matrix a2_pool = Max_Pool_Forward(a2, 2, 2);

	Shape a2p_shape = a2_pool.shape();
	a2_pool.reshape({ a2p_shape.d4, a2p_shape.d3 * a2p_shape.d2 * a2p_shape.d1 });
	
	Matrix a3 = a2_pool % weights.W[2];
	int cols = a3.shape().d1;
	int rows = a3.shape().d2;
	float* a3_ptr = a3.data();
	const float* b_ptr = weights.b[2].data();

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			a3_ptr[i * cols + j] += b_ptr[j];

	Relu_Inp(a3);
	
	Matrix z4 = a3 % weights.W[3] + weights.b[3](0, 0);
	Matrix hx = Sigmoid(z4);

	// Backward Pass //
	Matrix<float> dA_prev(hx.shape());
	const float m = hx.shape().d2;
	for (int i = 0; i < m; i++)
		dA_prev(i, 0) = -((Y(i, 0) / hx(i, 0)) - (1.0 - Y(i, 0)) / (1.0 - hx(i, 0)));

	Matrix dZL = Sigmoid_Gradient(z4) * dA_prev;
	w_grads.W[3] = (float)(1.0 / m) * (a3.Transpose() % dZL) + weights.W[3] * (lambda / m);
	w_grads.b[3](0,0) = (1.0 / m) * dZL.sum();

	dA_prev = dZL % weights.W[3].Transpose();
	Relu_Gradient_v2(a3);
	dZL = dA_prev * a3;
	w_grads.W[2] = (float)(1.0 / m) * (a2_pool.Transpose() % dZL) + weights.W[2] * (lambda / m);
	w_grads.b[2].set_Zero();
	rows = dZL.shape().d2;
	cols = dZL.shape().d1;
	float* b_grad_ptr = w_grads.b[2].data();
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			b_grad_ptr[j] += dZL(i, j);
	for (int i = 0; i < cols; i++)
		b_grad_ptr[i] /= m;

	dA_prev = dZL % weights.W[2].Transpose();
	dA_prev.reshape(a2p_shape);
	dZL = Max_Pool_Backward(a2, dA_prev, 2, 2);
	Relu_Gradient_v2(a2);
	dZL = a2 * dZL;
	if (!Convolve_Backward_v3(a1_pool, dZL, weights.W[1], w_grads.W[1], w_grads.b[1], dA_prev, 1)) {
		std::cerr << "\nError During Backward Convolution\n";
		return Matrix<float>();
	}
	w_grads.W[1] = w_grads.W[1] + weights.W[1] * (lambda / m);

	dZL = Max_Pool_Backward(a1, dA_prev, 2, 2);
	Relu_Gradient_v2(a1);
	dZL = a1 * dZL;
	if (!Convolve_Backward_v3(X, dZL, weights.W[0], w_grads.W[0], w_grads.b[0], dA_prev, 1)) {
		std::cerr << "\nError During Backward Convolution\n";
		return Matrix<float>();
	}
	w_grads.W[0] = w_grads.W[0] + weights.W[0] * (lambda / m);

	return hx;
}

