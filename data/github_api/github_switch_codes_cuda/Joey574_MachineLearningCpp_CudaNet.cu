// Repository: Joey574/MachineLearningCpp
// File: SingleBlockCudaNetwork/CudaNet.cu

#include "SingleBlockCudaNetwork.h"

void CudaNetwork::initialize_train_data(float** d_x_train, float** d_y_train, float** d_x_test, float** d_y_test, matrix h_x_train, matrix h_y_train, matrix h_x_test, matrix h_y_test) {
	// initialize training data on the gpu
	cudaMalloc(d_x_train, h_x_train.rows * h_x_train.cols * sizeof(float));
	cudaMalloc(d_y_train, h_y_train.rows * h_y_train.cols * sizeof(float));

	cudaMalloc(d_x_test, h_x_test.rows * h_x_test.cols * sizeof(float));
	cudaMalloc(d_y_test, h_y_test.rows * h_y_test.cols * sizeof(float));


	cudaMemcpy(*d_x_train, h_x_train.mat.data(), h_x_train.rows * h_x_train.cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_y_train, h_y_train.mat.data(), h_y_train.rows * h_y_train.cols * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(*d_x_test, h_x_test.mat.data(), h_x_test.rows * h_x_test.cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_y_test, h_y_test.mat.data(), h_y_test.rows * h_y_test.cols * sizeof(float), cudaMemcpyHostToDevice);
}
void CudaNetwork::initialize_batch_data(size_t batch_size) {
	m_batch_activation_size = 0;

	m_batch_data_size = m_network_size;

	for (size_t i = 1; i < m_dimensions.size(); i++) {
		m_batch_data_size += 3 * (m_dimensions[i] * batch_size);
		m_batch_activation_size += m_dimensions[i] * batch_size;
	}

	cudaMalloc(&m_batch_data, m_batch_data_size * sizeof(float));

	m_activation = &m_batch_data[m_batch_activation_size];

	m_d_total = &m_activation[m_batch_activation_size];
	m_d_weights = &m_d_total[m_batch_activation_size];
	m_d_bias = &m_d_weights[m_weights_size];

}
void CudaNetwork::initialize_test_data(size_t test_size) {
	m_test_activation_size = 0;

	for (size_t i = 1; i < m_dimensions.size(); i++) {
		m_test_activation_size += m_dimensions[i] * test_size;
	}

	cudaMalloc(&m_test_data, 2 * m_test_activation_size * sizeof(float));
	m_test_activation = &m_test_data[m_test_activation_size];
}

void CudaNetwork::define(std::vector<size_t> dimensions) {
	this->m_dimensions = dimensions;

	m_weights_size = 0;
	m_bias_size = 0;
	for (size_t i = 0; i < dimensions.size() - 1; i++) {
		m_weights_size += dimensions[i] * dimensions[i + 1];
		m_bias_size += dimensions[i + 1];
	}
	m_network_size = m_weights_size + m_bias_size;
}
void CudaNetwork::compile(CudaNetwork::weight_init init) {

	float* net = (float*)calloc(m_network_size, sizeof(float));

	std::random_device rd;
	std::default_random_engine gen(rd());

	size_t idx = 0;
	switch (init) {
	case weight_init::xavier: {
		for (size_t i = 0; i < m_dimensions.size() - 1; i++) {
			float lower_rand = -(1.0f / std::sqrt(m_dimensions[i + 1]));
			float upper_rand = 1.0f / std::sqrt(m_dimensions[i + 1]);

			std::uniform_real_distribution<float> dist_x(lower_rand, upper_rand);

			for (size_t j = 0; j < m_dimensions[i] * m_dimensions[i + 1]; j++, idx++) {
				net[idx] = dist_x(gen);
			}
		}
		break;
	}
	case weight_init::he: {
		float lower_rand = 0.0f;

		for (size_t i = 0; i < m_dimensions.size() - 1; i++) {
			float upper_rand = std::sqrt(2.0f / m_dimensions[i + 1]);

			std::normal_distribution<float> dist_h(lower_rand, upper_rand);

			for (size_t j = 0; j < m_dimensions[i] * m_dimensions[i + 1]; j++, idx++) {
				net[idx] = dist_h(gen);
			}
		}
		break;
	}
	case weight_init::normalize: {
		float lower_rand = -0.5f;
		float upper_rand = 0.5f;

		for (size_t i = 0; i < m_dimensions.size() - 1; i++) {
			std::uniform_real_distribution<float> dist_n(lower_rand, upper_rand);

			for (size_t j = 0; j < m_dimensions[i] * m_dimensions[i + 1]; j++, idx++) {
				net[idx] = dist_n(gen) * std::sqrt(1.0f / m_dimensions[i + 1]);
			}
		}
		break;
	}
	}

	cudaMalloc(&m_network, m_network_size * sizeof(float));
	cudaMemcpy(m_network, net, m_network_size * sizeof(float), cudaMemcpyHostToDevice);

	m_bias = m_network + m_weights_size;

	free(net);


	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "Status: network failed to compile: " << cudaGetErrorString(err) << "\n";
	} else {
		std::cout << "Status: network compiled\n";
	}
}
 
void CudaNetwork::fit(matrix x_train, matrix y_train, matrix x_test, matrix y_test, size_t batch_size, size_t epochs, float learning_rate, bool shuffle, int validation_freq) {
	std::cout << "Status: network_training\n";

	auto start_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time;

	float *d_x_train, *d_y_train, *d_x_test, *d_y_test;
	const size_t iterations = x_train.rows / batch_size;

	initialize_train_data(&d_x_train, &d_y_train, &d_x_test, &d_y_test, x_train, y_train, x_test, y_test);
	initialize_batch_data(batch_size);
	initialize_test_data(x_test.rows);

	std::cout << "\nPointers:\nd_x_train:\t" << d_x_train << "\nd_y_train:\t" << d_y_train << "\nd_x_test:\t" << d_x_test << "\nd_y_test:\t" << d_y_test << "\n";
	std::cout << "m_batch_data:\t" << m_batch_data << "\nm_test_data:\t" << m_test_data << "\n";

	for (size_t e = 0; e < epochs; e++) {
		auto epoch_start_time = std::chrono::high_resolution_clock::now();

		for (size_t i = 0; i < iterations; i++) {

			float* d_x = d_x_train + (i * batch_size * x_train.cols);
			float* d_y = d_y_train + (i * batch_size * y_train.cols);

			forward_prop(d_x, m_batch_data, m_batch_activation_size, batch_size);
			back_prop(d_x, d_y, learning_rate, batch_size);
		}
		std::cout << verbose(d_x_test, d_y_test, x_test.rows, e, validation_freq, epoch_start_time);
	}
	time = std::chrono::high_resolution_clock::now() - start_time;


	cudaFree(m_batch_data); m_batch_data = nullptr;
	cudaFree(m_test_data); m_test_data = nullptr;

	cudaFree(d_x_train);
	cudaFree(d_y_train);
	cudaFree(d_x_test);
	cudaFree(d_y_test);

	std::cout << "Status: training_complete\n";
}

void CudaNetwork::forward_prop(float* x_data, float* result_data, size_t activation_size, size_t num_elements) {

	if (num_elements == 10000) {
		std::cout << "FP Input:\n\trecieved x_data: " << x_data << "\n\trecieved result_data: " << result_data << "\n\trecieved activation_size: " << activation_size << "\n\trecieved num_elements: " << num_elements << "\n";
	}

	size_t weight_idx = 0;
	size_t bias_idx = 0;

	size_t input_idx = activation_size;
	size_t output_idx = 0;

	for (size_t i = 0; i < m_dimensions.size() - 1; i++) {

		dim3 grid(ceil((num_elements * m_dimensions[i + 1]) / 8));

		float* weights = m_network + weight_idx;
		float* bias = m_bias + bias_idx;

		float* input = i == 0 ? x_data : (result_data + input_idx);
		float* output = result_data + output_idx;

		float* activation = output + activation_size;

		// arguments
		void* dp_args[7] = { &weights, &input, &output, &m_dimensions[i + 1], &m_dimensions[i], (i == 0 ? &num_elements : &m_dimensions[i]), (i == 0 ? &m_dimensions[i] : &num_elements) };
		void* ba_args[4] = { &output, &bias, &m_dimensions[i + 1], &num_elements };
		void* af_args[4] = { &output, &activation, &m_dimensions[i + 1], &num_elements };

		// compute total
		i == 0 ? cudaLaunchKernel(dot_prod_t_b, grid, 8, dp_args, 0, nullptr) :
				 cudaLaunchKernel(dot_prod, grid, 8, dp_args, 0, nullptr);
		cudaDeviceSynchronize();
		
		// add bias
		cudaLaunchKernel(horizontal_add, ceil(m_dimensions[i + 1] / 8), 8, ba_args, 0, nullptr);
		cudaDeviceSynchronize();

		// activation funciton
		cudaLaunchKernel(leaky_relu, grid, 8, af_args, 0, nullptr);
		cudaDeviceSynchronize();

		weight_idx += m_dimensions[i] * m_dimensions[i + 1];
		bias_idx += m_dimensions[i + 1];

		input_idx += i == 0 ? 0 : (m_dimensions[i] * num_elements);
		output_idx += m_dimensions[i + 1] * num_elements;
	}
}
void CudaNetwork::back_prop(float* x_data, float* y_data, float learning_rate, size_t num_elements) {

	float factor = learning_rate / (float)num_elements;

	// -> compute loss
	{
		float* last_d_total = &m_d_total[m_batch_activation_size - (m_dimensions.back() * num_elements)];
		float* last_activation = &m_activation[m_batch_activation_size - (m_dimensions.back() * num_elements)];

		void* args[5] = { &last_activation, &last_d_total, &y_data, &m_dimensions.back(), &num_elements };
		
		cudaLaunchKernel(one_hot_loss, ceil(num_elements / 8), 8, args, 0, nullptr);
		cudaDeviceSynchronize();
	}


	// -> compute d_total
	{
		size_t weight_idx = m_weights_size - (m_dimensions.back() * m_dimensions[m_dimensions.size() - 2]);
		size_t d_total_idx = m_batch_activation_size - (m_dimensions.back() * num_elements);

		for (size_t i = m_dimensions.size() - 2; i > 0; i--) {

			float* weight = &m_network[weight_idx];
			float* prev_total = &m_batch_data[d_total_idx - (m_dimensions[i] * num_elements)];

			float* cur_d_total = &m_d_total[d_total_idx];
			float* prev_d_total = &m_d_total[d_total_idx - (m_dimensions[i] * num_elements)];

			void* dp_args[7] = { &weight, &cur_d_total, &prev_d_total, &m_dimensions[i + 1], &m_dimensions[i], &m_dimensions[i + 1], &num_elements };
			void* af_args[4] = { &prev_total, &prev_d_total, &m_dimensions[i], &num_elements };

			cudaLaunchKernel(dot_prod_t_a, ceil(m_dimensions[i + 1]), 8, dp_args, 0, nullptr);
			cudaDeviceSynchronize();

			// multiply by activation function derivative
			cudaLaunchKernel(leaky_relu_derivative, ceil(m_dimensions[i + 1] / 8), 8, af_args, 0, nullptr);
			cudaDeviceSynchronize();

			d_total_idx -= m_dimensions[i] * num_elements;
			weight_idx -= m_dimensions[i] * m_dimensions[i - 1];
		}
	}

	
	// -> compute d_weights and d_biases
	{
		size_t activation_idx = 0;

		size_t d_total_idx = 0;
		size_t d_weights_idx = 0;
		size_t d_bias_idx = 0;

		for (size_t i = 0; i < m_dimensions.size() - 1; i++) {

			dim3 grid(ceil((m_dimensions[i + 1] * m_dimensions[i]) / 8));

			float* prev_activation = i == 0 ? &x_data[0] : &m_activation[activation_idx];

			float* d_total = &m_d_total[d_total_idx];
			float* d_weights = &m_d_weights[d_weights_idx];
			float* d_bias = &m_d_bias[d_bias_idx];

			void* dw_args[7] = { &d_total, &prev_activation, &d_weights, &m_dimensions[i + 1], &num_elements, (i == 0 ? &num_elements : &m_dimensions[i]), (i == 0 ? &m_dimensions[i] : &num_elements) };
			void* db_args[4] = { &d_total, &d_bias, &m_dimensions[i + 1], &num_elements };

			// d_weights
			i == 0 ?
				cudaLaunchKernel(dot_prod, grid, 8, dw_args, 0, nullptr) :
				cudaLaunchKernel(dot_prod_t_b, grid, 8, dw_args, 0, nullptr);
			cudaDeviceSynchronize();

			// d_biases
			cudaLaunchKernel(horizontal_sum, ceil(m_dimensions[i + 1] / 8), 8, db_args, 0, nullptr);
			cudaDeviceSynchronize();

			d_bias_idx += m_dimensions[i + 1];
			d_total_idx += m_dimensions[i + 1] * num_elements;
			d_weights_idx += m_dimensions[i] * m_dimensions[i + 1];
			activation_idx += i == 0 ? 0 : (m_dimensions[i] * num_elements);
		}
	}

	// update weights and biases
	{
		void* uw_args[4] = { &m_network, &m_d_weights, &factor, &m_weights_size };
		void* ub_args[4] = { &m_bias, &m_d_bias, &factor, &m_bias_size };

		cudaLaunchKernel(update_weights, ceil(m_weights_size / 8), 8, uw_args, 0, nullptr);
		cudaLaunchKernel(update_bias, ceil(m_bias_size / 8), 8, ub_args, 0, nullptr);
		cudaDeviceSynchronize();
	}
}

std::string CudaNetwork::test_network(float* x, float* y, size_t test_size) {

	std::cout << "\nTest input:\n\tx: " << x << "\n\ty: " << y << "\n\ttest_size: " << test_size << "\n\txy dif: " << (x - y) << "\n";

	std::cout << "before test: " << cudaGetErrorString(cudaGetLastError()) << "\n";
	forward_prop(x, m_test_data, m_test_activation_size, test_size);
	std::cout << "after test: " << cudaGetErrorString(cudaGetLastError()) << "\n";

	int* d_correct;
	int correct = -1;

	cudaMalloc(&d_correct, sizeof(int));
	cudaMemset(d_correct, 0, sizeof(int));

	float* prediction = m_test_activation + m_test_activation_size - (m_dimensions.back() * test_size);

	void* args[5] = { &prediction, &y, &d_correct, &m_dimensions.back(), &test_size };

	cudaLaunchKernel(accuracy_score, ceil(test_size / 8), 7, args, 0, nullptr);

	cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	return "score: " + std::to_string(correct).append(" :: ").append(std::to_string(test_size)).append(" :: ").append(std::to_string((float)correct / (float)test_size));
	//return "score: " + std::to_string(((float)correct / (float)test_size) * 100.0f);
}
std::string CudaNetwork::verbose(float* d_x_test, float* d_y_test, size_t test_samples, size_t epoch, int validation_freq, std::chrono::steady_clock::time_point start_time) {
	std::string tmp = "Epoch: " + std::to_string(epoch).append(" Time: "); int tmp_len = tmp.length();
	if (epoch % validation_freq == 0) {
		tmp.append(test_network(d_x_test, d_y_test, test_samples));
	}
	const std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - start_time;

	return tmp.insert(tmp_len, clean_time(time.count()).append(" ")).append("\n");
}

std::string CudaNetwork::clean_time(double time) {
	const double hour = 3600000.00;
	const double minute = 60000.00;
	const double second = 1000.00;

	if (time / hour > 1.00) {
		return std::to_string(time / hour).append(" hours");
	} else if (time / minute > 1.00) {
		return std::to_string(time / minute).append(" minutes");
	} else if (time / second > 1.00) {
		return std::to_string(time / second).append(" seconds");
	} else {
		return std::to_string(time).append("(ms)");
	}
}