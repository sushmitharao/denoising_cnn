#pragma once

#include <vector>
#include <tuple>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include "utils.h"


class NoisyBSDSDataset : public torch::data::Dataset<NoisyBSDSDataset>
{
private:
	std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> img_list;
	size_t img_h;
	size_t img_w;
	bool mode;
	float sigma;

public:
	NoisyBSDSDataset(std::string& root_dir, bool isTrain, std::tuple<size_t, size_t> image_size,  int noise_var)
		// Load csv file with file names
		: img_list(ReadCSV(root_dir, mode)) {
		mode = isTrain;
		sigma = noise_var;
		std::tie(img_h, img_w) = image_size;
	};

	explicit NoisyBSDSDataset(std::string& root_dir, bool isTrain)
		// Load csv file with file names
		: img_list(ReadCSV(root_dir, mode)) {
		// Initialize parameters
		mode = isTrain;
		sigma = 30;
		std::tie(img_h, img_w) = (mode == 0) ? std::make_tuple(320,320) : std::make_tuple(180, 180);
	};

	// Override the get method to load custom data.
	torch::data::Example<> get(size_t index) override {

		std::string file_location = std::get<0>(img_list[index]);

		// Load image with OpenCV
		cv::Mat img = cv::imread(file_location);
		int offset_x = (rand() % (img.size().width - img_w)) + 1;
		int offset_y = (rand() % (img.size().height - img_h)) + 1;

		cv::Rect roi;
		roi.x = offset_x;
		roi.y = offset_y;
		roi.width = img_w;
		roi.height = img_h;

		// Crop the original image to the defined ROI to obtain clean image
		cv::Mat clean_img = img(roi);
		// Values in range 0-255, channels reordered to RGB
		cv::cvtColor(clean_img, clean_img, cv::COLOR_BGR2RGB);
		// 1. CV_32FC3 --> Converted to float type (0.0f - 255.0f); 2. 1.0f/255.0f --> Normalized to (0.0f - 1.0f)
		clean_img.convertTo(clean_img, CV_32FC3, 1.0f / 255.0f);
		std::vector<int64_t> dims = {clean_img.rows, clean_img.cols, clean_img.channels()};
		torch::Tensor img_tensor = torch::from_blob(clean_img.data, dims);
		// Scale and center the mean at 0. Range: (-1.0f,1.0f)
		img_tensor = 2 * img_tensor - 1;

		// Reshape tensor so that it is of form - Channels * Height * Width (was H *  W * C before permute)
		torch::Tensor clean_img_tensor = img_tensor.permute({2,0,1});
		// Generate the noisy image to be used as input to the network
		torch::Tensor noisy_img_tensor = (clean_img_tensor + (1.0f / 255.0f) * sigma * torch::randn(clean_img_tensor.sizes()));
		// Noisy image - Input; Clean image - Target
		return { noisy_img_tensor, clean_img_tensor };
	};

	// Override the size method to infer the size of the dataset
	torch::optional<size_t> size() const override {

		return img_list.size();
	};
};

