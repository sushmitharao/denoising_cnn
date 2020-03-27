#pragma once

#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// Additional headers
#include <sstream> 
#include <fstream>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils.h"
#include "NoisyBSDSDataset.h"
#include "globals.h"
#include "dncnn_model.h"



/* Train function loads dataset in batches and trains through the DnCNN model architecture
   using Adam optimization to minimize mean squared loss */
template <typename DataLoader>
void train(
	int32_t epoch,
	Net& model,
	torch::Device device,
	DataLoader& data_loader,
	torch::optim::Optimizer& optimizer,
	size_t dataset_size) 
{
	model.train();
	size_t batch_idx = 0;
	double train_loss = 0;
	double psnr = 0;
	double b_idx = 0;

	
	for (auto& batch : data_loader) 
	{

		auto data = batch.data.to(device), targets = batch.target.to(device);
		optimizer.zero_grad();
		auto output = model.forward(data);
		auto n = data.numel();
		auto loss = torch::mse_loss(output, targets);
		train_loss += loss.item<double>();
		auto psnr_temp = 10 * torch::log10((4 * n) / torch::norm((output - targets), 2).pow(2));
		psnr = psnr + psnr_temp.item<double>();

		loss.backward();
		optimizer.step();
		
		if (batch_idx++ % kLogInterval == 0) {
			std::cout << "Running train_loss is " << train_loss << std::endl;
			if (b_idx != 0)
			{
				std::printf(
					"\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f Loss wrt updates: %.4f \n",
					epoch,
					batch_idx * batch.data.size(0),
					dataset_size,
					loss.template item<float>(), train_loss / b_idx);
			}
		}
		b_idx+=1;
	}
	train_loss = train_loss / batch_idx;
	psnr = psnr / batch_idx;
	std::printf(
		"\nTrain Epoch: %ld MSE loss: %.4lf, PSNR: %.4lf \n",
		epoch, train_loss, psnr);
	output_log << "train: " << epoch << "," << train_loss <<","<< psnr << std::endl;
}

/* Test function loads test images in batches and forward propagates through the DnCNN model architecture
   and calls save output function, displays performance metrics - mse loss, psnr */
template <typename DataLoader>
void test(
	Net& model,
	torch::Device device,
	DataLoader& data_loader,
	size_t dataset_size)
{
	torch::NoGradGuard no_grad;
	model.eval();
	double test_loss = 0;
	double psnr = 0;
	size_t batch_idx = 0;

	for (auto& batch : data_loader) 
	{
		auto data = batch.data.to(device), targets = batch.target.to(device);
		auto output = model.forward(data);
		SaveOutput(data, targets, output, kTestBatchSize);

		auto n = data.numel();

		auto loss = torch::mse_loss(output, targets);
		auto psnr_temp = 10 * torch::log10((4 * n) / torch::norm((output - targets),2).pow(2));
		test_loss = test_loss + loss.item<double>();
		psnr = psnr + psnr_temp.item<double>();
		batch_idx++;
	}

	test_loss = test_loss/batch_idx;
	psnr = psnr/batch_idx;
	std::printf(
		"\nTest set: MSE loss: %.4lf, PSNR: %.4lf \n",
		test_loss, psnr);
	output_log << "test: " << test_loss << "," << psnr << std::endl;
}

auto main() -> int {

	torch::manual_seed(1);

	torch::DeviceType device_type;

	// Create the device we pass around based on whether CUDA is available
	torch::Device device(torch::kCPU);

	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Training on GPU." << std::endl;
		cudaDeviceProp cuda_dev;
		for (int w = 0; w < torch::cuda::device_count(); w++) {
			cudaGetDeviceProperties(&cuda_dev, w);
			printf("device id = %d, device name = %s\n", w, cuda_dev.name);
		}

		int id = 0;
		device = torch::Device(torch::kCUDA, id);
		cudaGetDeviceProperties(&cuda_dev, id);
		printf("use device id = %d, device name = %s\n\n", id, cuda_dev.name);
	}
	else {
		std::cout << "Training on CPU." << std::endl;
	}

	// Instantiate NoisyBSDS Dataset for training dataset
	auto train_dataset = NoisyBSDSDataset(kDataRoot, TRAIN, std::make_tuple(180,180), 30)
		.map(torch::data::transforms::Stack<>());

	// Get number of training images
	const size_t train_dataset_size = train_dataset.size().value();

	// Load training images in batches
	auto train_loader =
		torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));

	// Instantiate NoisyBSDS Dataset for training dataset
	auto test_dataset = NoisyBSDSDataset(
		kDataRoot, TEST)
		.map(torch::data::transforms::Stack<>());

	// Get number of test images
	const size_t test_dataset_size = test_dataset.size().value();

	// Load test dataset in batches 
	auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

	Net model;
	// If CUDA available, move model to CUDA
	model.to(device);

	// Using Adam optimizer - learning rate = 1e-3
	torch::optim::Adam optimizer(
		model.parameters(), torch::optim::AdamOptions(1e-3));

	output_log.open("../../output_folder/output_log.txt");
	// Train the model
	for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
		std::cout << "Training..." << std::endl;
		train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
	}
	
	// Forward pass through trained model and obtain denoised images, performance metrics
	std::cout << "Testing..." << std::endl;
	test(model, device, *test_loader, test_dataset_size);
	std::cout << "Completed testing!" << std::endl;
	output_log.close();
}


