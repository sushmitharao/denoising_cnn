#pragma once

#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>

// Read in the root_dir, mode and return file name as a vector .
auto ReadCSV(std::string& location, bool mode) -> std::vector<std::tuple<std::string, int64_t>> {

    std::string line;
    std::string name;
    //std::vector<std::string> csv;
	std::vector<std::tuple<std::string, int64_t>> csv;
    std::string file_names_csv = location;

    file_names_csv.append((mode == 0)?"train/list.csv":"test/list.csv");
    std::fstream input(file_names_csv, std::ios::in);

    while (getline(input, line))
    {
        name = location;
        name.append((mode == 0)?"train/":"test/");
        name.append(line);
		csv.push_back(std::make_tuple(name, 0));
    }

    return csv;
}

/* Saves clean, noisy and output for each test image input */
void SaveOutput(const torch::Tensor &data, const torch::Tensor &targets, const torch::Tensor &output, size_t batch_size)
{
	static int idx = 0;

	for (int i = 0; i < batch_size; i++)
	{
		// Save output image
		auto ntensor = output.slice(0, i, i + 1, 1).squeeze(0).detach().permute({ 1, 2, 0 });
		auto sizes = ntensor.sizes();

		ntensor = 0.5 * (ntensor + 1);
		ntensor = ntensor.mul(255).clamp(0, 255).to(torch::kU8);
		ntensor = ntensor.to(torch::kCPU);
		cv::Mat resultImg(180, 180, CV_8UC3);
		std::memcpy((void *)resultImg.data, ntensor.data_ptr(), sizeof(torch::kU8) * ntensor.numel());

		cv::Mat output_image = resultImg;
		cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR);
		std::string ofilename = "../../output_folder/output_" + std::to_string(idx) + ".jpg";
		cv::imwrite(ofilename, output_image);

		// Save clean (target) image
		auto ntensor1 = targets.slice(0, i, i + 1, 1).squeeze(0).detach().permute({ 1, 2, 0 });
		auto sizes1 = ntensor1.sizes();

		ntensor1 = 0.5 * (ntensor1 + 1);
		ntensor1 = ntensor1.mul(255).clamp(0, 255).to(torch::kU8);
		ntensor1 = ntensor1.to(torch::kCPU);

		cv::Mat resultImg1(180, 180, CV_8UC3);
		std::memcpy((void *)resultImg1.data, ntensor1.data_ptr(), sizeof(torch::kU8) * ntensor1.numel());

		cv::Mat clean_image = resultImg1;
		cv::cvtColor(clean_image, clean_image, cv::COLOR_RGB2BGR);
		std::string cfilename = "../../output_folder/clean_" + std::to_string(idx) + ".jpg";

		cv::imwrite(cfilename, clean_image);

		// Save noisy (input/data) image
		auto ntensor2 = data.slice(0, i, i + 1, 1).squeeze(0).detach().permute({ 1, 2, 0 });
		auto sizes2 = ntensor2.sizes();

		ntensor2 = 0.5 * (ntensor2 + 1);
		ntensor2 = ntensor2.mul(255).clamp(0, 255).to(torch::kU8);
		ntensor2 = ntensor2.to(torch::kCPU);

		cv::Mat resultImg2(180, 180, CV_8UC3);

		std::memcpy((void *)resultImg2.data, ntensor2.data_ptr(), sizeof(torch::kU8) * ntensor2.numel());

		cv::Mat noisy_image = resultImg2;
		std::string nfilename = "../../output_folder/noisy_" + std::to_string(idx) + ".jpg";
		cv::cvtColor(noisy_image, noisy_image, cv::COLOR_RGB2BGR);
		cv::imwrite(nfilename, noisy_image);

		idx++;
	}
}


