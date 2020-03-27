#pragma once

struct Net : torch::nn::Module
{
	Net()
		:conv0(torch::nn::Conv2dOptions(3, 64, 3).padding(1)),
		conv1(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
		conv2(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
		conv3(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
		conv4(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
		conv5(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
		conv6(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
		conv7(torch::nn::Conv2dOptions(64, 3, 3).padding(1)),

		bn0(torch::nn::BatchNorm2d(64)),
		bn1(torch::nn::BatchNorm2d(64)),
		bn2(torch::nn::BatchNorm2d(64)),
		bn3(torch::nn::BatchNorm2d(64)),
		bn4(torch::nn::BatchNorm2d(64)),
		bn5(torch::nn::BatchNorm2d(64))

	{
		register_module("conv0", conv0);
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("conv4", conv4);
		register_module("conv5", conv5);
		register_module("conv6", conv6);
		register_module("conv7", conv7);

		register_module("bn0", bn0);
		register_module("bn1", bn1);
		register_module("bn2", bn2);
		register_module("bn3", bn3);
		register_module("bn4", bn4);
		register_module("bn5", bn5);
	}

	torch::Tensor forward(torch::Tensor input)
	{
		//std::cout << "inside forward function" << std::endl;
		//if (MODE == TRAIN)	std::cout << input.sizes() << input.type() << std::endl;
		//if (MODE == TEST)	std::cout << input/*input.sizes() << input.type() */<< std::endl;

		auto h = torch::relu(conv0(input));

		//if (MODE == TEST)	std::cout << "testt 0" << std::endl;

		h = torch::relu(bn0(conv1(h)));
		//if (MODE == TEST)	std::cout << "testt 1" << std::endl;
		h = torch::relu(bn1(conv2(h)));
		h = torch::relu(bn2(conv3(h)));
		h = torch::relu(bn3(conv4(h)));
		h = torch::relu(bn4(conv5(h)));
		h = torch::relu(bn5(conv6(h)));
		//if (MODE == TEST)	std::cout << "testt 2" << std::endl;

		h = conv7(h) + input;
		//std::cout << "returning input" << input << std::endl;
		//if (MODE == TEST)	std::cout << "testt 3" << std::endl;
		return h;
	}
	torch::nn::Conv2d conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7;
	torch::nn::BatchNorm2d bn0, bn1, bn2, bn3, bn4, bn5;

};

