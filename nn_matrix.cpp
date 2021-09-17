#include "nn_matrix.h"

void error_not_2D_Mat(const char* msg, const Shape& s)
{
	printf("\nError from function: %s Matrix is not a 2D Matrix, shape is : (%d,%d,%d,%d)\nAborting...", msg, s.d4, s.d3, s.d2, s.d1);
	abort();
}

void error_shape_mismatch(const Shape& s1, const Shape& s2, const char* msg)
{
	printf("\nError form function: %s, Shapes of Matrices dont matches : (%d,%d,%d,%d) and (%d,%d,%d,%d)\nAborting...",
		msg, s1.d4, s1.d3, s1.d2, s1.d1, s2.d4, s2.d3, s2.d2, s2.d1);
	abort();
}

template<>
void Matrix<float>::set_Random(const float mean, const float sigma) {
	VSLStreamStatePtr rn_stream;
	vslNewStream(&rn_stream, VSL_BRNG_MT19937, m_id);
	vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rn_stream, size(), m_data, mean, sigma);
}

template<>
void Matrix<double>::set_Random(const double mean, const double sigma) {
	VSLStreamStatePtr rn_stream;
	vslNewStream(&rn_stream, VSL_BRNG_MT19937, m_id);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rn_stream, size(), m_data, mean, sigma);
}

template<>
void Matrix<float>::inplace_Transpose()
{
	if (m_shape.m_type != M_Type::MAT_2D)error_not_2D_Mat("Inp_Transpose", m_shape);

	mkl_simatcopy('R', 'T', m_shape.d2, m_shape.d1, 1.0, m_data, m_shape.d1, m_shape.d2);
	std::swap(m_shape.d2, m_shape.d1);
}

template<>
void Matrix<double>::inplace_Transpose()
{
	if (m_shape.m_type != M_Type::MAT_2D)error_not_2D_Mat("Inp_Transpose", m_shape);

	mkl_dimatcopy('R', 'T', m_shape.d2, m_shape.d1, 1.0, m_data, m_shape.d1, m_shape.d2);
	std::swap(m_shape.d2, m_shape.d1);
}

template<>
Matrix<float> Matrix<float>::Transpose() const 
{
	if (m_shape.m_type != M_Type::MAT_2D)error_not_2D_Mat("Transpose", m_shape);

	Matrix<float> tmp({ m_shape.d1,m_shape.d2 });
	mkl_somatcopy('R', 'T', m_shape.d2, m_shape.d1, 1.0, m_data, m_shape.d1, tmp.data(), m_shape.d2);
	return tmp;
}

template<>
Matrix<double> Matrix<double>::Transpose() const 
{
	if (m_shape.m_type != M_Type::MAT_2D)error_not_2D_Mat("Transpose", m_shape);

	Matrix<double> tmp({ m_shape.d1,m_shape.d2 });
	mkl_domatcopy('R', 'T', m_shape.d2, m_shape.d1, 1.0, m_data, m_shape.d1, tmp.data(), m_shape.d2);
	return tmp;
}

template<>
Matrix<int> Matrix<int>::Transpose() const
{
	if (m_shape.m_type != M_Type::MAT_2D)error_not_2D_Mat("Transpose", m_shape);
	int m = m_shape.d2;
	int n = m_shape.d1;
	Matrix<int> tmp({ n,m });

	int* tmp_data = tmp.data();
	int block_size = (n >= 2000) ? 32 : 16;
	int ii = 0, jj = 0, cb = 0, rb = 0;

	for (rb = 0; rb <= m - block_size; rb += block_size) {
		for (cb = 0; cb <= n - block_size; cb += block_size) {
			for (ii = rb; ii < rb + block_size; ii++)
				for (jj = cb; jj < cb + block_size; jj++)
					tmp_data[jj * m + ii] = m_data[ii * n + jj];
		}
		for (ii = rb; ii < rb + block_size; ii++)
			for (jj = cb; jj < n; jj++)
				tmp_data[jj * m + ii] = m_data[ii * n + jj];
	}

	for (cb = 0; cb <= n - block_size; cb += block_size) {
		for (ii = rb; ii < m; ii++)
			for (jj = cb; jj < cb + block_size; jj++)
				tmp_data[jj * m + ii] = m_data[ii * n + jj];
	}
	for (ii = rb; ii < m; ii++)
		for (jj = cb; jj < n; jj++)
			tmp_data[jj * m + ii] = m_data[ii * n + jj];

	return tmp;
}

template<>
void Matrix<float>::inp_abs() // To be tested , for Correctness ///
{
	vsAbs(size(), m_data, m_data);
}

template<>
void Matrix<double>::inp_abs()  // To be tested , for Correctness ///
{
	vdAbs(size(), m_data, m_data);
}

template<>
void Matrix<int>::inp_abs() 
{
	size_t n = size();
	for(size_t i = 0; i<n; i++)
		m_data[i] = (m_data[i] > 0) ? m_data[i] : (-m_data[i]);
}

template<>
Matrix<float> Matrix<float>::abs() const 
{
	Matrix<float> tmp(m_shape);
	vsAbs(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::abs() const 
{
	Matrix<double> tmp(m_shape);
	vdAbs(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<int> Matrix<int>::abs() const 
{
	Matrix<int> tmp(m_shape);
	size_t n = size();
	int* tmp_ptr = tmp.data();

	for (size_t i = 0; i < n; i++)
		tmp_ptr[i] = (m_data[i] > 0) ? m_data[i] : (-m_data[i]);

	return tmp;
}

template<>
Matrix<float> Matrix<float>::element_Inverse() const 
{
	Matrix<float> tmp({ m_shape });
	vsInv(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::element_Inverse() const
{
	Matrix<double> tmp({ m_shape });
	vdInv(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<float> Matrix<float>::square() const
{
	Matrix<float> tmp({ m_shape });
	vsSqr(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::square() const
{
	Matrix<double> tmp({ m_shape });
	vdSqr(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<int> Matrix<int>::square() const
{
	Matrix<int> tmp(m_shape);
	size_t n = size();
	int* tmp_ptr = tmp.data();

	for (int i = 0; i < n; i++)
		tmp_ptr[i] = m_data[i] * m_data[i];

	return tmp;
}

template<>
Matrix<float> Matrix<float>::sqrt() const
{
	Matrix<float> tmp({ m_shape });
	vsSqrt(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::sqrt() const
{
	Matrix<double> tmp({ m_shape });
	vdSqrt(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<int> Matrix<int>::sqrt() const
{
	Matrix<int> tmp(m_shape);
	size_t n = size();
	int* tmp_ptr = tmp.data();

	for (int i = 0; i < n; i++)
		tmp_ptr[i] = (int)sqrtf(m_data[i]);

	return tmp;
}

template<>
Matrix<float> Matrix<float>::invSqrt() const
{
	Matrix<float> tmp({ m_shape });
	vsInvSqrt(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::invSqrt() const
{
	Matrix<double> tmp({ m_shape });
	vdInvSqrt(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<float> Matrix<float>::cbrt() const
{
	Matrix<float> tmp({ m_shape });
	vsCbrt(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::cbrt() const
{
	Matrix<double> tmp({ m_shape });
	vdCbrt(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<float> Matrix<float>::pow(const float val) const
{
	Matrix<float> tmp({ m_shape });
	vsPowx(size(), m_data, val, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::pow(const double val) const
{
	Matrix<double> tmp({ m_shape });
	vdPowx(size(), m_data, val, tmp.data());
	return tmp;
}

template<>
Matrix<float> Matrix<float>::log_e() const
{
	Matrix<float> tmp({ m_shape });
	vsLn(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::log_e() const
{
	Matrix<double> tmp({ m_shape });
	vdLn(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<float> Matrix<float>::log_10() const
{
	Matrix<float> tmp({ m_shape });
	vsLog10(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::log_10() const
{
	Matrix<double> tmp({ m_shape });
	vdLog10(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<float> Matrix<float>::exp() const
{
	Matrix<float> tmp({ m_shape });
	vsExp(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::exp() const
{
	Matrix<double> tmp({ m_shape });
	vdExp(size(), m_data, tmp.data()); 
	return tmp;
}

template<>
Matrix<float> Matrix<float>::sin() const
{
	Matrix<float> tmp({ m_shape });
	vsSin(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::sin() const
{
	Matrix<double> tmp({ m_shape });
	vdSin(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<float> Matrix<float>::cos() const
{
	Matrix<float> tmp({ m_shape });
	vsCos(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::cos() const
{
	Matrix<double> tmp({ m_shape });
	vdCos(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<float> Matrix<float>::tan() const
{
	Matrix<float> tmp({ m_shape });
	vsTan(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::tan() const
{
	Matrix<double> tmp({ m_shape });
	vdTan(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<float> Matrix<float>::tanh() const
{
	Matrix<float> tmp({ m_shape });
	vsTanh(size(), m_data, tmp.data());
	return tmp;
}

template<>
Matrix<double> Matrix<double>::tanh() const
{
	Matrix<double> tmp({ m_shape });
	vdTanh(size(), m_data, tmp.data());
	return tmp;
}

template<>
float Matrix<float>::dot(const Matrix<float>& other) const
{
	if (!(m_shape == other.m_shape))error_shape_mismatch(m_shape, other.m_shape, "dot product");

	float res = cblas_sdsdot(size(), 0, m_data, 1, other.m_data, 1);
	return res;
}

template<>
double Matrix<double>::dot(const Matrix<double>& other) const
{
	if (!(m_shape == other.m_shape))error_shape_mismatch(m_shape, other.m_shape, "dot product");

	double res = cblas_ddot(size(), m_data, 1, other.m_data, 1);
	return res;
}

template<>
float Matrix<float>::sum() const
{
	size_t n = size();
	__m256 t_acc = _mm256_load_ps(m_data);
	size_t i = 0;
	for (i = 8; i <= n - 8; i += 8)
		t_acc = _mm256_add_ps(_mm256_load_ps(&m_data[i]), t_acc);

	float tmp_sum_1 = 0, tmp_sum_2 = 0;
	for (int rmd = i; rmd < n; rmd++)
		tmp_sum_1 += m_data[rmd];

	for (int j = 0; j < 8; j++)
		tmp_sum_2 += t_acc.m256_f32[j];

	return (tmp_sum_1 + tmp_sum_2);
}

template<>
double Matrix<double>::sum() const
{
	size_t n = size();
	__m256d t_acc = _mm256_load_pd(m_data);
	size_t i = 0;
	for (i = 4; i <= n - 4; i += 4)
		t_acc = _mm256_add_pd(_mm256_load_pd(&m_data[i]), t_acc);

	double tmp_sum_1 = 0, tmp_sum_2 = 0;
	for (int rmd = i; rmd < n; rmd++)
		tmp_sum_1 += m_data[rmd];

	tmp_sum_2 = (t_acc.m256d_f64[0] + t_acc.m256d_f64[1] + t_acc.m256d_f64[2] + t_acc.m256d_f64[3]);

	return (tmp_sum_1 + tmp_sum_2);
}

template<>
float Matrix<float>::mean() const
{
	return sum() / size();
}

template<>
double Matrix<double>::mean() const
{
	return sum() / size();
}

template<>
float Matrix<float>::max() const
{
	size_t n = size();
	__m256 max = _mm256_load_ps(m_data);
	size_t i = 0;
	for (i = 8; i <= n - 8; i += 8)
		max = _mm256_max_ps(max, _mm256_load_ps(&m_data[i]));

	float tmp_max_1 = m_data[i];
	for (int rmd = i + 1; rmd < n; rmd++)
		if (tmp_max_1 < m_data[rmd])
			tmp_max_1 = m_data[rmd];

	float tmp_max_2 = max.m256_f32[0];
	for (int j = 1; j < 8; j++)
		if (tmp_max_2 < max.m256_f32[j])
			tmp_max_2 = max.m256_f32[j];

	return std::max(tmp_max_1, tmp_max_2);
}

template<>
double Matrix<double>::max() const
{
	size_t n = size();
	__m256d max = _mm256_load_pd(m_data);
	size_t i = 0;
	for (i = 4; i <= n - 4; i += 4)
		max = _mm256_max_pd(max, _mm256_load_pd(&m_data[i]));

	double tmp_max_1 = m_data[i];
	for (int rmd = i + 1; rmd < n; rmd++)
		if (tmp_max_1 < m_data[rmd])
			tmp_max_1 = m_data[rmd];

	double tmp_max_2 = max.m256d_f64[0];
	for (int j = 1; j < 4; j++)
		if (tmp_max_2 < max.m256d_f64[j])
			tmp_max_2 = max.m256d_f64[j];

	return std::max(tmp_max_1, tmp_max_2);
}

template<>
float Matrix<float>::min() const
{
	size_t n = size();
	__m256 min = _mm256_load_ps(m_data);
	size_t i = 0;
	for (i = 8; i <= n - 8; i += 8)
		min = _mm256_min_ps(min, _mm256_load_ps(&m_data[i]));

	float tmp_min_1 = m_data[i];
	for (int rmd = i + 1; rmd < n; rmd++)
		if (tmp_min_1 > m_data[rmd])
			tmp_min_1 = m_data[rmd];

	float tmp_min_2 = min.m256_f32[0];
	for (int j = 1; j < 8; j++)
		if (tmp_min_2 > min.m256_f32[j])
			tmp_min_2 = min.m256_f32[j];

	return std::min(tmp_min_1, tmp_min_2);
}

template<>
double Matrix<double>::min() const
{
	size_t n = size();
	__m256d min = _mm256_load_pd(m_data);
	size_t i = 0;
	for (i = 4; i <= n - 4; i += 4)
		min = _mm256_min_pd(min, _mm256_load_pd(&m_data[i]));

	double tmp_min_1 = m_data[i];
	for (int rmd = i + 1; rmd < n; rmd++)
		if (tmp_min_1 > m_data[rmd])
			tmp_min_1 = m_data[rmd];

	double tmp_min_2 = min.m256d_f64[0];
	for (int j = 1; j < 4; j++)
		if (tmp_min_2 > min.m256d_f64[j])
			tmp_min_2 = min.m256d_f64[j];

	return std::min(tmp_min_1, tmp_min_2);
}

Matrix<float> operator+(Matrix<float>& mat_1, Matrix<float>& mat_2)
{
	if (!(mat_1.shape() == mat_2.shape()))error_shape_mismatch(mat_1.shape(), mat_2.shape(), "operator +");

	Matrix<float> tmp(mat_1.shape());
	vsAdd(mat_1.size(), mat_1.data(), mat_2.data(), tmp.data());
	return tmp;
}

Matrix<double> operator+(Matrix<double>& mat_1, Matrix<double>& mat_2)
{
	if (!(mat_1.shape() == mat_2.shape()))error_shape_mismatch(mat_1.shape(), mat_2.shape(), "operator +");

	Matrix<double> tmp(mat_1.shape());
	vdAdd(mat_1.size(), mat_1.data(), mat_2.data(), tmp.data());
	return tmp;
}

Matrix<int> operator+(Matrix<int>& mat_1, Matrix<int>& mat_2)
{
	if (!(mat_1.shape() == mat_2.shape()))error_shape_mismatch(mat_1.shape(), mat_2.shape(), "operator +");

	Matrix<int> tmp(mat_1.shape());
	const int* m1 = mat_1.data();
	const int* m2 = mat_2.data();
	int* res = tmp.data();
	size_t n = mat_1.size();
	
	for (int i = 0; i < n; i++)
		res[i] = m1[i] + m2[i];

	return tmp;
}

Matrix<float> operator-(Matrix<float>& mat_1, Matrix<float>& mat_2)
{
	if (!(mat_1.shape() == mat_2.shape()))error_shape_mismatch(mat_1.shape(), mat_2.shape(), "operator -");

	Matrix<float> tmp(mat_1.shape());
	vsSub(mat_1.size(), mat_1.data(), mat_2.data(), tmp.data());
	return tmp;
}

Matrix<double> operator-(Matrix<double>& mat_1, Matrix<double>& mat_2)
{
	if (!(mat_1.shape() == mat_2.shape()))error_shape_mismatch(mat_1.shape(), mat_2.shape(), "operator -");

	Matrix<double> tmp(mat_1.shape());
	vdSub(mat_1.size(), mat_1.data(), mat_2.data(), tmp.data());
	return tmp;
}

Matrix<int> operator-(Matrix<int>& mat_1, Matrix<int>& mat_2)
{
	if (!(mat_1.shape() == mat_2.shape()))error_shape_mismatch(mat_1.shape(), mat_2.shape(), "operator -");

	Matrix<int> tmp(mat_1.shape());
	const int* m1 = mat_1.data();
	const int* m2 = mat_2.data();
	int* res = tmp.data();
	size_t n = mat_1.size();

	for (int i = 0; i < n; i++)
		res[i] = m1[i] - m2[i];

	return tmp;
}

Matrix<float> operator%(Matrix<float>& mat_1, Matrix<float>& mat_2)  // To be Tested for, Correctness //
{
	Shape m1_s = mat_1.shape();
	Shape m2_s = mat_2.shape();
	if (m1_s.m_type != M_Type::MAT_2D)error_not_2D_Mat("Matrix Multiplication", m1_s);
	if (m2_s.m_type != M_Type::MAT_2D)error_not_2D_Mat("Matrix Multiplication", m2_s);

	int m = m1_s.d2;
	int k = m1_s.d1;
	int n = m2_s.d1;
	if (k != m2_s.d2) {
		std::cout << "\nError Invalid shape Matrices for Matrix Multiplication\nAborting...";
		abort();
	}

	Matrix<float> tmp({ m,n });

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k, 1.0, mat_1.data(), k, mat_2.data(), n, 0.0, tmp.data(), n);
	
	return tmp;
}

Matrix<double> operator%(Matrix<double>& mat_1, Matrix<double>& mat_2)  // To be Tested for, Correctness //
{
	Shape m1_s = mat_1.shape();
	Shape m2_s = mat_2.shape();
	if (m1_s.m_type != M_Type::MAT_2D)error_not_2D_Mat("Matrix Multiplication", m1_s);
	if (m2_s.m_type != M_Type::MAT_2D)error_not_2D_Mat("Matrix Multiplication", m2_s);

	int m = m1_s.d2;
	int k = m1_s.d1;
	int n = m2_s.d1;
	if (k != m2_s.d2) {
		std::cout << "\nError Invalid shape Matrices for Matrix Multiplication\nAborting...";
		abort();
	}

	Matrix<double> tmp({ m,n });

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k, 1.0, mat_1.data(), k, mat_2.data(), n, 0.0, tmp.data(), n);

	return tmp;
}

Matrix<float> operator*(const Matrix<float>& mat_1, const Matrix<float>& mat_2)
{
	if (!(mat_1.shape() == mat_2.shape()))error_shape_mismatch(mat_1.shape(), mat_2.shape(), "operator *");

	Matrix<float> tmp(mat_1.shape());
	vsMul(mat_1.size(), mat_1.data(), mat_2.data(), tmp.data());
	return tmp;
}

Matrix<double> operator*(const Matrix<double>& mat_1, const Matrix<double>& mat_2)
{
	if (!(mat_1.shape() == mat_2.shape()))error_shape_mismatch(mat_1.shape(), mat_2.shape(), "operator *");

	Matrix<double> tmp(mat_1.shape());
	vdMul(mat_1.size(), mat_1.data(), mat_2.data(), tmp.data());
	return tmp;
}

Matrix<int> operator*(const Matrix<int>& mat_1, const Matrix<int>& mat_2)
{
	if (!(mat_1.shape() == mat_2.shape()))error_shape_mismatch(mat_1.shape(), mat_2.shape(), "operator *");

	Matrix<int> tmp(mat_1.shape());
	const int* m1 = mat_1.data();
	const int* m2 = mat_2.data();
	int* res = tmp.data();
	size_t n = mat_1.size();

	for (int i = 0; i < n; i++)
		res[i] = m1[i] * m2[i];

	return tmp;
}

Matrix<float> operator/(Matrix<float>& mat_1, Matrix<float>& mat_2)
{
	if (!(mat_1.shape() == mat_2.shape()))error_shape_mismatch(mat_1.shape(), mat_2.shape(), "operator /");

	Matrix<float> tmp(mat_1.shape());
	vsDiv(mat_1.size(), mat_1.data(), mat_2.data(), tmp.data());
	return tmp;
}

Matrix<double> operator/(Matrix<double>& mat_1, Matrix<double>& mat_2)
{
	if (!(mat_1.shape() == mat_2.shape()))error_shape_mismatch(mat_1.shape(), mat_2.shape(), "operator /");

	Matrix<double> tmp(mat_1.shape());
	vdDiv(mat_1.size(), mat_1.data(), mat_2.data(), tmp.data());
	return tmp;
}

Matrix<int> operator/(Matrix<int>& mat_1, Matrix<int>& mat_2)
{
	if (!(mat_1.shape() == mat_2.shape()))error_shape_mismatch(mat_1.shape(), mat_2.shape(), "operator /");

	Matrix<int> tmp(mat_1.shape());
	const int* m1 = mat_1.data();
	const int* m2 = mat_2.data();
	int* res = tmp.data();
	size_t n = mat_1.size();

	for (int i = 0; i < n; i++)
		res[i] = m1[i] / m2[i];

	return tmp;
}