#ifndef MLP_HPP
#define MLP_HPP


#include <iostream>
#include <array> // Added for  idsp::c_vector
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <type_traits>

#include "idsp_addons.hpp"



template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t OUT_SIZE>
class MLP_3LYR {
public:
    MLP_3LYR(size_t epochs);
    idsp::c_vector<double, OUT_SIZE> forward(const idsp::c_vector<double, IN_SIZE>& inputs, bool linear_output = false);
    void train(
            const idsp::c_vector<idsp::c_vector<double, IN_SIZE>, 99>& inputs, 
            const idsp::c_vector<idsp::c_vector<double, OUT_SIZE>, 99>& targets, 
            double learning_rate, int epochs
            );
    idsp::c_vector<double, OUT_SIZE> generate_output();
    void fill_input_vec(int idx, double f) { this->net_in[idx] = f; this->generate_output(); }
    void clear_data();
    void accumulate_data(const idsp::c_vector<double, IN_SIZE>& inputs, const idsp::c_vector<double, OUT_SIZE>& targets);
    void retrain(double learning_rate, int epochs);
    void set_training_amount(size_t amount) { training_epochs = amount; }
    idsp::c_vector<double, IN_SIZE> find_closest_data_set(const idsp::c_vector<double, OUT_SIZE>& target);

private:
    size_t training_epochs;

    idsp::c_vector<double, IN_SIZE> net_in;

    idsp::c_vector<idsp::c_vector<double, H1_SIZE>, IN_SIZE> weight_in_h1;
    idsp::c_vector<idsp::c_vector<double, H2_SIZE>, H1_SIZE> weight_h1_h2;
    idsp::c_vector<idsp::c_vector<double, H3_SIZE>, H2_SIZE> weight_h2_h3;
    idsp::c_vector<idsp::c_vector<double, OUT_SIZE>, H3_SIZE> weight_h3_out;

    idsp::c_vector<double, H1_SIZE> bias_h1;
    idsp::c_vector<double, H2_SIZE> bias_h2;
    idsp::c_vector<double, H3_SIZE> bias_h3;
    idsp::c_vector<double, OUT_SIZE> bias_out;

    idsp::c_vector<idsp::c_vector<double, IN_SIZE>, 99> accumulated_inputs;
    idsp::c_vector<idsp::c_vector<double, OUT_SIZE>, 99> accumulated_targets;
    size_t accumulated_data_size;

    void initialize_weights_and_biases();

    // Activation functions
    double tanh_activation(double x) {
        return std::tanh(x);
    }

    double tanh_derivative(double x) {
        return 1.0 - std::tanh(x) * std::tanh(x);
    }
    
    double get_random_bipolar_normalised() {
        return ((double)rand() / RAND_MAX) * 2 - 1;
    }

    // Mean Squared Error loss function
    double mean_squared_error(const  idsp::c_vector<double, 6>& predicted, const  idsp::c_vector<double, 6>& target) {  
        double error = 0.0;
        for (size_t i = 0; i < predicted.size(); ++i) {
            error += (predicted[i] - target[i]) * (predicted[i] - target[i]);
        }
        return error / predicted.size();
    }
};

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t H4_SIZE, size_t OUT_SIZE>
class MLP_4LYR {
public:
    MLP_4LYR(size_t epochs);
    idsp::c_vector<double, OUT_SIZE> forward(const idsp::c_vector<double, IN_SIZE>& inputs, bool linear_output = false);
    void train(
            const idsp::c_vector<idsp::c_vector<double, IN_SIZE>, 99>& inputs, 
            const idsp::c_vector<idsp::c_vector<double, OUT_SIZE>, 99>& targets, 
            double learning_rate, int epochs
            );
    idsp::c_vector<double, OUT_SIZE> generate_output();
    void fill_input_vec(int idx, double f) { this->net_in[idx] = f; this->generate_output(); }
    void clear_data();
    void accumulate_data(const idsp::c_vector<double, IN_SIZE>& inputs, const idsp::c_vector<double, OUT_SIZE>& targets);
    void retrain(double learning_rate, int epochs);
    void set_training_amount(size_t amount) { training_epochs = amount; }
    idsp::c_vector<double, IN_SIZE> find_closest_data_set(const idsp::c_vector<double, OUT_SIZE>& target);

private:
    size_t training_epochs;

    idsp::c_vector<double, IN_SIZE> net_in;

    idsp::c_vector<idsp::c_vector<double, H1_SIZE>, IN_SIZE> weight_in_h1;
    idsp::c_vector<idsp::c_vector<double, H2_SIZE>, H1_SIZE> weight_h1_h2;
    idsp::c_vector<idsp::c_vector<double, H3_SIZE>, H2_SIZE> weight_h2_h3;
    idsp::c_vector<idsp::c_vector<double, H4_SIZE>, H3_SIZE> weight_h3_h4;
    idsp::c_vector<idsp::c_vector<double, OUT_SIZE>, H4_SIZE> weight_h4_out;

    idsp::c_vector<double, H1_SIZE> bias_h1;
    idsp::c_vector<double, H2_SIZE> bias_h2;
    idsp::c_vector<double, H3_SIZE> bias_h3;
    idsp::c_vector<double, H4_SIZE> bias_h4;
    idsp::c_vector<double, OUT_SIZE> bias_out;

    idsp::c_vector<idsp::c_vector<double, IN_SIZE>, 99> accumulated_inputs;
    idsp::c_vector<idsp::c_vector<double, OUT_SIZE>, 99> accumulated_targets;
    size_t accumulated_data_size;

    void initialize_weights_and_biases();

    // Activation functions
    double tanh_activation(double x) {
        return std::tanh(x);
    }

    double tanh_derivative(double x) {
        return 1.0 - std::tanh(x) * std::tanh(x);
    }

    double get_random_bipolar_normalised() {
        return ((double)rand() / RAND_MAX) * 2 - 1;
    }

    // Mean Squared Error loss function
    double mean_squared_error(const  idsp::c_vector<double, 6>& predicted, const  idsp::c_vector<double, 6>& target) {  
        double error = 0.0;
        for (size_t i = 0; i < predicted.size(); ++i) {
            error += (predicted[i] - target[i]) * (predicted[i] - target[i]);
        }
        return error / predicted.size();
    }
};

// Include the template implementation file
#include "mlp.tpp"

#endif // MLP_HPP