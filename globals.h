#pragma once
#include <fstream>
// May be required in NoisyBSDSDataset class
enum modes { TRAIN = 0, TEST = 1 };

// Where to find the BSDS dataset
std::string kDataRoot = "../../data/images/";

// The batch size for training.
const int64_t kTrainBatchSize = 5;

// The batch size for testing.
const int64_t kTestBatchSize = 25;

// The number of epochs to train. 
const int64_t kNumberOfEpochs = 2;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

// Output log file
std::ofstream output_log;