#pragma once
#include <iostream>
#include <immintrin.h>
#include <mkl.h>
#include <omp.h>

#define NUM_THREADS 4

enum class M_Type : uint8_t {
	MAT_1D = 1,
	MAT_2D,
	MAT_3D,
	MAT_4D
};

struct Shape {
	int d4 = 1,
		d3 = 1,
		d2 = 1,
		d1 = 1;
	M_Type m_type = M_Type::MAT_1D;

	Shape(int d_4, int d_3, int d_2, int d_1) {
		d1 = d_1; d2 = d_2;
		d3 = d_3; d4 = d_4;
		m_type = M_Type::MAT_4D;
	}

	Shape(int d_3, int d_2, int d_1) {
		d1 = d_1; d2 = d_2; d3 = d_3;
		m_type = M_Type::MAT_3D;
	}

	Shape(int d_2, int d_1) {
		d1 = d_1; d2 = d_2;
		m_type = M_Type::MAT_2D;
	}

	Shape(int d_1) {
		d1 = d_1;
		m_type = M_Type::MAT_1D;
	}
	Shape() {}

	void print() const {
		std::cout << '(' << d4 << ' ' << d3 << ' ' << d2 << ' ' << d1 << ')' << ':' << (int)m_type << '\n';
	}
};

inline bool operator==(const Shape& lhs, const Shape& rhs) {
	return (lhs.d1 == rhs.d1 && lhs.d2 == rhs.d2 && lhs.d3 == rhs.d3 && lhs.d4 == rhs.d4);
}

void error_not_2D_Mat(const char* msg, const Shape& s);
void error_shape_mismatch(const Shape& s1, const Shape& s2, const char* msg);

template<typename T>
class Matrix {

private:
	T* m_data;
	int m_id;
	Shape m_shape;

public:

	Matrix(const Shape shape) {
		m_shape.d1 = shape.d1 < 1 ? 1 : shape.d1;
		m_shape.d2 = shape.d2 < 1 ? 1 : shape.d2;
		m_shape.d3 = shape.d3 < 1 ? 1 : shape.d3;
		m_shape.d4 = shape.d4 < 1 ? 1 : shape.d4;
		m_shape.m_type = shape.m_type;

		m_data = (T*)mkl_malloc(sizeof(T) * m_shape.d4 * m_shape.d3 * m_shape.d2 * m_shape.d1, 64);
		if (m_data) {
			m_id = rand();
		}
		else {
			this->~Matrix();
			std::cerr << "\nError Occured while creating Object\n";
			abort();
		}
		//std::cout << "Object Created id: " << m_id << '\n';
	}

	Matrix() :m_data(nullptr) {
		m_id = rand(); //std::cout << "Object Created id: " << m_id << '\n'; 
	}

	Matrix(const Matrix<T>& other) :m_shape(other.m_shape) {
		//std::cout << "Copy Constructor Called\n";
		m_data = (T*)mkl_malloc(sizeof(T) * m_shape.d4 * m_shape.d3 * m_shape.d2 * m_shape.d1, 64);
		memcpy(m_data, other.m_data, sizeof(T) * m_shape.d4 * m_shape.d3 * m_shape.d2 * m_shape.d1);
		m_id = rand();
	}

	Matrix(Matrix<T>&& other) noexcept :m_data(nullptr) {
		//std::cout << "Move Constructor Called\n";
		m_data = other.m_data;
		m_shape = other.m_shape;
		m_id = other.m_id;

		other.m_data = nullptr;
	}

	void print() const {
		for (int n = 0; n < m_shape.d4; n++) {
			std::cout << '[';
			for (int z = 0; z < m_shape.d3; z++) {
				std::cout << '[';
				for (int y = 0; y < m_shape.d2; y++) {
					std::cout << "[ ";
					for (int x = 0; x < m_shape.d1; x++) 
						std::cout << m_data[n * m_shape.d3 * m_shape.d2 * m_shape.d1 + z * m_shape.d2 * m_shape.d1 + y * m_shape.d1 + x] << ' ';
					std::cout << " ]\n";
				}
				std::cout << "]\n";
			}
			std::cout << "]\n";
		}
		printf("Shape : (%d,%d,%d,%d)\n", m_shape.d4, m_shape.d3, m_shape.d2, m_shape.d1);
		//printf("Size : %d  Shape : (%d,%d,%d,%d) dtype : %s id: %d\n", size(), m_shape.d4, m_shape.d3, m_shape.d2, m_shape.d1, typeid(T).name(), m_id);
	}

	void set_Zero() {
		memset(m_data, 0, sizeof(T) * m_shape.d4 * m_shape.d3 * m_shape.d2 * m_shape.d1);
	}

	void set_Constant(const T val) {
		size_t n = (size_t)m_shape.d4 * m_shape.d3 * m_shape.d2 * m_shape.d1;
		for (size_t i = 0; i < n; i++)
			m_data[i] = val;
	}

	void iota(T val) {
		size_t n = (size_t)m_shape.d4 * m_shape.d3 * m_shape.d2 * m_shape.d1;
		for (size_t i = 0; i < n; i++)
			m_data[i] = val++;
	}

	void set_Random(const T mean = (T)0.0, const T sigma = (T)1.0);

	T* data() { return m_data; }
	T* data() const { return m_data; }
	Shape shape() const { return m_shape; }
	size_t size() const { return (size_t)m_shape.d4 * m_shape.d3 * m_shape.d2 * m_shape.d1; }

	void reshape(const Shape shape) {
		if (size() != (size_t)shape.d1 * shape.d2 * shape.d3 * shape.d4) {
			std::cerr << "\nSize dont match Cannot Reshape to new Shape Aborting...\n";
			abort();
		}
		else m_shape = shape;
	}

	void resize(const Shape shape) {
		if (size() != (size_t)shape.d1 * shape.d2 * shape.d3 * shape.d4) {
			mkl_free(m_data);
			m_data = (T*)mkl_malloc(sizeof(T) * shape.d1 * shape.d2 * shape.d3 * shape.d4, 64);
		}
		m_shape = shape;
	}

	Matrix<T> block(int d4_srt, int d4_stp, int d3_srt, int d3_stp, int d2_srt, int d2_stp, int d1_srt, int d1_stp) const; 

	Matrix<T>& operator=(const Matrix<T>& other) {
		//std::cout << "\nAssignment operator called\n";
		if (this != &other) {
			if (size() != other.size()) {
				mkl_free(m_data);
				m_data = (T*)mkl_malloc(sizeof(T) * other.size(), 64);
			}
			m_shape = other.m_shape;
			memcpy(m_data, other.m_data, sizeof(T) * other.size());
		}
		return *this;
	}

	Matrix<T>& operator=(Matrix<T>&& other) noexcept {
		//std::cout << "Move Assignment operator Called\n";
		if (this != &other) {
			mkl_free(m_data);

			m_data = other.m_data;
			m_shape = other.m_shape;

			other.m_data = nullptr;
		}
		return *this;
	}

	T& operator()(int d1);
	T& operator()(int d2, int d1);
	T& operator()(int d3, int d2, int d1);
	T& operator()(int d4, int d3, int d2, int d1);

	Matrix<T> operator-() const;  // - unary operator ///

	Matrix<T> abs() const;
	void inp_abs();
	Matrix<T> element_Inverse() const;
	Matrix<T> square() const;
	Matrix<T> sqrt() const;
	Matrix<T> invSqrt() const;
	Matrix<T> cbrt() const;
	Matrix<T> pow(const T val) const;
	Matrix<T> log_e() const;
	Matrix<T> log_10() const;
	Matrix<T> exp() const;
	Matrix<T> sin() const;
	Matrix<T> cos() const;
	Matrix<T> tan() const;
	Matrix<T> tanh() const;
	T dot(const Matrix<T>& other) const;
	T sum() const;
	T mean() const;
	T max() const;
	T min() const;

	/*
	* Only Implemented for 2D Matrices only
	* Behaviour undefined otherwise
	*/
	void inplace_Transpose();
	Matrix<T> Transpose() const;

	~Matrix() {
		//std::cout << "Destructor called of mat id: " << m_id << '\n';
		mkl_free(m_data);
	}
};

template<typename T>
inline T& Matrix<T>::operator()(int d1)
{
	return m_data[d1];
}

template<typename T>
inline T& Matrix<T>::operator()(int d2, int d1)
{
	return m_data[m_shape.d1 * d2 + d1];
}

template<typename T>
inline T& Matrix<T>::operator()(int d3, int d2, int d1)
{
	return m_data[m_shape.d2 * m_shape.d1 * d3 + m_shape.d1 * d2 + d1];
}

template<typename T>
inline T& Matrix<T>::operator()(int d4, int d3, int d2, int d1)
{
	return m_data[m_shape.d3 * m_shape.d2 * m_shape.d1 * d4 + m_shape.d2 * m_shape.d1 * d3 + m_shape.d1 * d2 + d1];
}

template<typename T>
Matrix<T> Matrix<T>::operator-() const
{
	Matrix<T> tmp(m_shape);
	size_t n = size();
	for (size_t i = 0; i < n; i++)
		tmp.m_data[i] = -m_data[i];
	return tmp;
}

template<typename T>
Matrix<T> Matrix<T>::block(int d4_srt, int d4_stp, int d3_srt, int d3_stp, int d2_srt, int d2_stp, int d1_srt, int d1_stp) const 
{
	int t_d4 = d4_stp - d4_srt + 1;
	int t_d3 = d3_stp - d3_srt + 1;
	int t_d2 = d2_stp - d2_srt + 1;
	int t_d1 = d1_stp - d1_srt + 1;
	Matrix<T> tmp({ t_d4, t_d3, t_d2, t_d1 });

	for (int n = 0; n < t_d4; n++)
		for (int z = 0; z < t_d3; z++)
			for (int y = 0; y < t_d2; y++)
				memcpy(&tmp.m_data[n * t_d3 * t_d2 * t_d1 + z * t_d2 * t_d1 + y * t_d1],
					&m_data[(d4_srt + n) * m_shape.d3 * m_shape.d2 * m_shape.d1 + (d3_srt + z) * m_shape.d2 * m_shape.d1 + (d2_srt + y) * m_shape.d1 + d1_srt],
					sizeof(T) * t_d1);

	return tmp;
}

Matrix<float> operator+(Matrix<float>& mat_1, Matrix<float>& mat_2);
Matrix<double> operator+(Matrix<double>& mat_1, Matrix<double>& mat_2);
Matrix<int> operator+(Matrix<int>& mat_1, Matrix<int>& mat_2);

Matrix<float> operator-(Matrix<float>& mat_1, Matrix<float>& mat_2);
Matrix<double> operator-(Matrix<double>& mat_1, Matrix<double>& mat_2);
Matrix<int> operator-(Matrix<int>& mat_1, Matrix<int>& mat_2);

Matrix<float> operator*(const Matrix<float>& mat_1, const Matrix<float>& mat_2);
Matrix<double> operator*(const Matrix<double>& mat_1, const Matrix<double>& mat_2);
Matrix<int> operator*(const Matrix<int>& mat_1, const Matrix<int>& mat_2);

Matrix<float> operator%(Matrix<float>& mat_1, Matrix<float>& mat_2);
Matrix<double> operator%(Matrix<double>& mat_1, Matrix<double>& mat_2);

Matrix<float> operator/(Matrix<float>& mat_1, Matrix<float>& mat_2);
Matrix<double> operator/(Matrix<double>& mat_1, Matrix<double>& mat_2);
Matrix<int> operator/(Matrix<int>& mat_1, Matrix<int>& mat_2);

template<typename T>
inline Matrix<T> operator+(Matrix<T>& mat, const T scalar)
{
	Matrix<T> tmp(mat.shape());
	size_t n = mat.size();
	T* mat_ptr = mat.data();
	T* tmp_ptr = tmp.data();
	for (size_t i = 0; i < n; i++)
		tmp_ptr[i] = mat_ptr[i] + scalar;

	return tmp;
}

template<typename T>
inline Matrix<T> operator+(const T scalar, Matrix<T>& mat)
{
	return (mat + scalar);
}

template<typename T>
inline Matrix<T> operator-(Matrix<T>& mat, const T scalar)
{
	Matrix<T> tmp(mat.shape());
	size_t n = mat.size();
	T* mat_ptr = mat.data();
	T* tmp_ptr = tmp.data();
	for (size_t i = 0; i < n; i++)
		tmp_ptr[i] = mat_ptr[i] - scalar;

	return tmp;
}

template<typename T>
inline Matrix<T> operator*(Matrix<T>& mat, const T scalar)
{
	Matrix<T> tmp(mat.shape());
	size_t n = mat.size();
	T* mat_ptr = mat.data();
	T* tmp_ptr = tmp.data();
	for (size_t i = 0; i < n; i++)
		tmp_ptr[i] = mat_ptr[i] * scalar;

	return tmp;
}

template<typename T>
inline Matrix<T> operator*(const T scalar, Matrix<T>& mat)
{
	return (mat * scalar);
}

template<typename T>
inline Matrix<T> operator/(Matrix<T>& mat, const T scalar)
{
	Matrix<T> tmp(mat.shape());
	size_t n = mat.size();
	T* mat_ptr = mat.data();
	T* tmp_ptr = tmp.data();
	for (size_t i = 0; i < n; i++)
		tmp_ptr[i] = mat_ptr[i] / scalar;

	return tmp;
}
