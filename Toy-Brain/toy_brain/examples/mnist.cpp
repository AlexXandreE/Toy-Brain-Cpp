#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/models.h"

using namespace std;
using namespace ToyBrain;

/**
 * Assuming data from: https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data
 */

namespace {

uint32_t read_be_u32(ifstream& in) {
	unsigned char bytes[4] = {0, 0, 0, 0};
	in.read(reinterpret_cast<char*>(bytes), 4);
	if (!in) {
		throw runtime_error("Unexpected end of file while reading header");
	}
	return (static_cast<uint32_t>(bytes[0]) << 24) |
		(static_cast<uint32_t>(bytes[1]) << 16) |
		(static_cast<uint32_t>(bytes[2]) << 8) |
		 static_cast<uint32_t>(bytes[3]);
}

vector<vector<double>> load_mnist_images(const string& path, size_t limit = 0) {
	ifstream in(path, ios::binary);
	if (!in) {
		throw runtime_error("Could not open image file: " + path);
	}

	const uint32_t magic = read_be_u32(in);
	if (magic != 2051) {
		throw runtime_error("Invalid image magic number in: " + path);
	}

	const uint32_t count = read_be_u32(in);
	const uint32_t rows = read_be_u32(in);
	const uint32_t cols = read_be_u32(in);
	const size_t image_size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
	const size_t to_read = (limit == 0) ? static_cast<size_t>(count) : min(static_cast<size_t>(count), limit);

	vector<vector<double>> images;
	images.reserve(to_read);

	vector<unsigned char> buffer(image_size);
	for (size_t i = 0; i < to_read; ++i) {
		in.read(reinterpret_cast<char*>(buffer.data()), static_cast<streamsize>(image_size));
		if (!in) {
			throw runtime_error("Unexpected end of image data in: " + path);
		}

		vector<double> normalized(image_size);
		for (size_t p = 0; p < image_size; ++p) {
			normalized[p] = static_cast<double>(buffer[p]) / 255.0;
		}
		images.push_back(std::move(normalized));
	}

	return images;
}

vector<uint8_t> load_mnist_labels(const string& path, size_t limit = 0) {
	ifstream in(path, ios::binary);
	if (!in) {
		throw runtime_error("Could not open label file: " + path);
	}

	const uint32_t magic = read_be_u32(in);
	if (magic != 2049) {
		throw runtime_error("Invalid label magic number in: " + path);
	}

	const uint32_t count = read_be_u32(in);
	const size_t to_read = (limit == 0) ? static_cast<size_t>(count) : min(static_cast<size_t>(count), limit);

	vector<uint8_t> labels(to_read);
	in.read(reinterpret_cast<char*>(labels.data()), static_cast<streamsize>(to_read));
	if (!in) {
		throw runtime_error("Unexpected end of label data in: " + path);
	}

	return labels;
}

void save_image_as_pgm(const string& file_path, const vector<double>& image, size_t rows, size_t cols) {
	if (image.size() != rows * cols) {
		throw runtime_error("Image size mismatch while exporting: " + file_path);
	}

	ofstream out(file_path, ios::binary);
	if (!out) {
		throw runtime_error("Could not create output image: " + file_path);
	}

	out << "P5\n" << cols << " " << rows << "\n255\n";
	for (double pixel : image) {
		double clamped = max(0.0, min(1.0, pixel));
		unsigned char value = static_cast<unsigned char>(clamped * 255.0);
		out.write(reinterpret_cast<const char*>(&value), 1);
	}
}

} // namespace

int main(int argc, char** argv) {
	const string train_images = "data/mnist/train-images.idx3-ubyte";
	const string train_labels = "data/mnist/train-labels.idx1-ubyte";
	const string test_images = "data/mnist/t10k-images.idx3-ubyte";
	const string test_labels = "data/mnist/t10k-labels.idx1-ubyte";
	
	// The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
	const size_t train_limit = 60000;
	const size_t test_limit = 10000;

	const string export_dir = (argc > 1) ? argv[1] : "";
    const bool export_all = (argc > 2) ? (string(argv[2]) == "1") : false;

	try {
		cout << "Loading MNIST..." << endl;
		auto x_train = load_mnist_images(train_images, train_limit);
		auto y_train = load_mnist_labels(train_labels, train_limit);
		auto x_test = load_mnist_images(test_images, test_limit);
		auto y_test = load_mnist_labels(test_labels, test_limit);

		if (x_train.empty() || x_test.empty()) {
			throw runtime_error("MNIST dataset is empty");
		}
		if (x_train.size() != y_train.size()) {
			throw runtime_error("Train images/labels count mismatch");
		}
		if (x_test.size() != y_test.size()) {
			throw runtime_error("Test images/labels count mismatch");
		}

		if (!export_dir.empty()) {
			std::filesystem::create_directories(export_dir);
		}

		const size_t input_size = x_train[0].size();
		const int hidden_size = 128;
		MultiLayerPerceptron model({static_cast<int>(input_size), hidden_size, hidden_size, 10}, Function::sigmoid);

		const int epochs = 10;
		const double learning_rate = 0.125;
		std::mt19937 rng(42);
		std::vector<size_t> shuffled_indices(x_train.size());
		std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);

		cout << "Training multiclass classifier: digits 0-9 (3-layer MLP)" << endl;
		for (int epoch = 0; epoch < epochs; ++epoch) {
			std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), rng);

			size_t correct = 0;
			vector<size_t> class_seen(10, 0);
			vector<size_t> class_correct(10, 0);
			double epoch_loss = 0.0;

			for (size_t order_index = 0; order_index < shuffled_indices.size(); ++order_index) {
				const size_t i = shuffled_indices[order_index];
				const int label = static_cast<int>(y_train[i]);
				const int predicted = model.predict(x_train[i]);

				++class_seen[static_cast<size_t>(label)];
				if (predicted == label) {
					++correct;
					++class_correct[static_cast<size_t>(label)];
				}

				epoch_loss += model.train_sample(x_train[i], y_train[i], learning_rate);
			}

			const double train_acc = static_cast<double>(correct) / static_cast<double>(x_train.size());
			cout << "Epoch " << (epoch + 1) << "/" << epochs
				 << " - train accuracy: " << fixed << setprecision(4) << train_acc
				 << " - loss: " << fixed << setprecision(4)
				 << (epoch_loss / static_cast<double>(x_train.size()));

			cout << " - per-class: ";
			for (size_t digit = 0; digit < 10; ++digit) {
				const double class_acc = (class_seen[digit] == 0)
					? 0.0
					: static_cast<double>(class_correct[digit]) / static_cast<double>(class_seen[digit]);
				cout << digit << ":" << setprecision(2) << class_acc;
				if (digit < 9) {
					cout << " ";
				}
			}
			cout << setprecision(4) << endl;
		}

		size_t test_correct = 0;
		vector<vector<int>> confusion(10, vector<int>(10, 0));
		for (size_t i = 0; i < x_test.size(); ++i) {
			const int label = static_cast<int>(y_test[i]);
			const int predicted = model.predict(x_test[i]);
			if (predicted == label) {
				++test_correct;
			}
			confusion[static_cast<size_t>(label)][static_cast<size_t>(predicted)]++;

			if (!export_dir.empty() && (export_all || predicted != label)) {
				const string file_name =
					"img_" + to_string(i) +
					"_true_" + to_string(label) +
					"_pred_" + to_string(predicted) +
					".pgm";
				save_image_as_pgm(export_dir + "/" + file_name, x_test[i], 28, 28);
			}
		}

		const double test_acc = static_cast<double>(test_correct) / static_cast<double>(x_test.size());
		cout << "Test multiclass accuracy: " << fixed << setprecision(4) << test_acc << endl;

		cout << "Confusion matrix (rows=true label, cols=pred):" << endl;
		for (size_t row = 0; row < 10; ++row) {
			for (size_t col = 0; col < 10; ++col) {
				cout << setw(4) << confusion[row][col];
			}
			cout << endl;
		}

		if (!export_dir.empty()) {
			cout << "Exported test images to: " << export_dir << endl;
			cout << "Usage: ./mnist [export_dir] [export_all]" << endl;
		}
	}
	catch (const exception& e) {
		cerr << "MNIST example failed: " << e.what() << endl;
		return 1;
	}

	return 0;
}
