// #include "mlp_tanh.hpp"

// // Activation functions
// double tanh_activation(double x) {
//     return std::tanh(x);
// }

// double tanh_derivative(double x) {
//     return 1.0 - std::tanh(x) * std::tanh(x);
// }

// // Mean Squared Error loss function
// double mean_squared_error(const  idsp::c_vector<double, 6>& predicted, const  idsp::c_vector<double, 6>& target) {  
//     double error = 0.0;
//     for (size_t i = 0; i < predicted.size(); ++i) {
//         error += (predicted[i] - target[i]) * (predicted[i] - target[i]);
//     }
//     return error / predicted.size();
// }

// // MLP_TANH class implementation
// MLP_TANH::MLP_TANH(size_t in, size_t h1, size_t h2, size_t h3, size_t out, size_t epochs) : 
// in_size(in), 
// h1_size(h1), 
// h2_size(h2), 
// h3_size(h3), 
// out_size(out),
// training_epochs(epochs),
// accumulated_data_size(0)
// {
//     initialize_weights_and_biases();
//      idsp::c_vector< idsp::c_vector<double, 3>, 7> inputs = {
//          idsp::c_vector<double, 3>{0., 0., 0.},
//          idsp::c_vector<double, 3>{1., 0., 0.},
//          idsp::c_vector<double, 3>{0., 1., 0.},
//          idsp::c_vector<double, 3>{0., 0., 1.},
//          idsp::c_vector<double, 3>{0.5, 0.2, 0.8},
//          idsp::c_vector<double, 3>{0.1, 0.4, 0.3},
//          idsp::c_vector<double, 3>{0.7, 0.2, 0.6}
//     };

//      idsp::c_vector< idsp::c_vector<double, 6>, 7> targets = {
//          idsp::c_vector<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
//          idsp::c_vector<double, 6>{0.9, -0.9, 0.0, 0.0, 0.0, 0.0},
//          idsp::c_vector<double, 6>{0.0, 0.0, 0.9, -0.9, 0.0, 0.0},
//          idsp::c_vector<double, 6>{0.0, 0.0, 0.0, 0.0, 0.9, -0.9},
//          idsp::c_vector<double, 6>{0.6, -0.4, 0.7, -0.3, 0.2, 0.5},
//          idsp::c_vector<double, 6>{-0.1, 0.3, -0.6, 0.4, -0.5, 0.8},
//          idsp::c_vector<double, 6>{0.2, -0.3, 0.4, -0.5, 0.6, 0.7}
//     };

//     train(inputs, targets, 0.01, training_epochs);
// }

//  idsp::c_vector<double, 6> MLP_TANH::forward(const  idsp::c_vector<double, 3>& current_input, bool linear_output) {  
//     if (current_input.size() != in_size) {
//         throw std::invalid_argument("Input size does not match");
//     }

//     // Hidden layer 1
//      idsp::c_vector<double, 10> hidden1;// = {0.0}; 
//     //  hidden1.fill(0.0); 
//     for (int i = 0; i < h1_size; ++i) {
//         for (int j = 0; j < in_size; ++j) {
//             hidden1[i] += current_input[j] * weight_in_h1[j][i];
//         }
//         hidden1[i] += bias_h1[i];
//         hidden1[i] = tanh_activation(hidden1[i]);
//     }

//     // Hidden layer 2
//      idsp::c_vector<double, 8> hidden2;// = {0.0};  
//         // hidden2.fill(0.0);
//     for (int i = 0; i < h2_size; ++i) {
//         for (int j = 0; j < h1_size; ++j) {
//             hidden2[i] += hidden1[j] * weight_h1_h2[j][i];
//         }
//         hidden2[i] += bias_h2[i];
//         hidden2[i] = tanh_activation(hidden2[i]);
//     }

//     // Hidden layer 3
//      idsp::c_vector<double, 10> hidden3;// = {0.0};  
//     //  hidden3.fill(0.0);
//     for (int i = 0; i < h3_size; ++i) {
//         for (int j = 0; j < h2_size; ++j) {
//             hidden3[i] += hidden2[j] * weight_h2_h3[j][i];
//         }
//         hidden3[i] += bias_h3[i];
//         hidden3[i] = tanh_activation(hidden3[i]);
//     }

//     // Output layer
//      idsp::c_vector<double, 6> outputs;// = {0.0};  
//         // outputs.fill(0.0);
//     for (int i = 0; i < out_size; ++i) {
//         for (int j = 0; j < h3_size; ++j) {
//             outputs[i] += hidden3[j] * weight_h3_out[j][i];
//         }
//         outputs[i] += bias_out[i];
//         outputs[i] = linear_output ? outputs[i] : tanh_activation(outputs[i]);
//     }

//     return outputs;
// }

// void MLP_TANH::train(const  idsp::c_vector< idsp::c_vector<double, 3>, 7>& saved_inputs, const  idsp::c_vector< idsp::c_vector<double, 6>, 7>& targets, double learning_rate, int epochs) {  
//     if (saved_inputs.size() != targets.size()) {
//         throw std::invalid_argument("Inputs and targets must have the same size");
//     }
//     std::cout << "Training MLP with " << saved_inputs.size() << " data sets\n";

//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         double total_loss = 0.0;

//         for (size_t i = 0; i < saved_inputs.size(); ++i) {
//             // Forward pass
//              idsp::c_vector<double, 10> hidden1;// = {0.0};
//             //  hidden1.fill(0.0);
//              idsp::c_vector<double, 8> hidden2;// = {0.0};
//             //  hidden2.fill(0.0);
//              idsp::c_vector<double, 10> hidden3;// = {0.0};
//             //  hidden3.fill(0.0);
//              idsp::c_vector<double, 6> outputs;// = {0.0};
//             //  outputs.fill(0.0);

//             // Hidden layer 1
//             for (int j = 0; j < h1_size; ++j) {
//                 for (int k = 0; k < in_size; ++k) {
//                     hidden1[j] += saved_inputs[i][k] * weight_in_h1[k][j];
//                 }
//                 hidden1[j] += bias_h1[j];
//                 hidden1[j] = tanh_activation(hidden1[j]);
//             }

//             // Hidden layer 2
//             for (int j = 0; j < h2_size; ++j) {
//                 for (int k = 0; k < h1_size; ++k) {
//                     hidden2[j] += hidden1[k] * weight_h1_h2[k][j];
//                 }
//                 hidden2[j] += bias_h2[j];
//                 hidden2[j] = tanh_activation(hidden2[j]);
//             }

//             // Hidden layer 3
//             for (int j = 0; j < h3_size; ++j) {
//                 for (int k = 0; k < h2_size; ++k) {
//                     hidden3[j] += hidden2[k] * weight_h2_h3[k][j];
//                 }
//                 hidden3[j] += bias_h3[j];
//                 hidden3[j] = tanh_activation(hidden3[j]);
//             }

//             // Output layer
//             for (int j = 0; j < out_size; ++j) {
//                 for (int k = 0; k < h3_size; ++k) {
//                     outputs[j] += hidden3[k] * weight_h3_out[k][j];
//                 }
//                 outputs[j] += bias_out[j];
//                 outputs[j] = tanh_activation(outputs[j]);
//             }

//             // Compute loss
//             total_loss += mean_squared_error(outputs, targets[i]);

//             // Backpropagation
//              idsp::c_vector<double, 6> output_errors;// = {0.0};  
//             //  output_errors.fill(0.0);
//             for (int j = 0; j < out_size; ++j) {
//                 output_errors[j] = (targets[i][j] - outputs[j]) * tanh_derivative(outputs[j]);
//             }

//              idsp::c_vector<double, 10> h3_errors;// = {0.0};  
//             //  h3_errors.fill(0.0);
//             for (int j = 0; j < h3_size; ++j) {
//                 for (int k = 0; k < out_size; ++k) {
//                     h3_errors[j] += output_errors[k] * weight_h3_out[j][k];
//                 }
//                 h3_errors[j] *= tanh_derivative(hidden3[j]);
//             }

//              idsp::c_vector<double, 8> h2_errors;// = {0.0};  
//                 // h2_errors.fill(0.0);
//             for (int j = 0; j < h2_size; ++j) {
//                 for (int k = 0; k < h3_size; ++k) {
//                     h2_errors[j] += h3_errors[k] * weight_h2_h3[j][k];
//                 }
//                 h2_errors[j] *= tanh_derivative(hidden2[j]);
//             }

//              idsp::c_vector<double, 10> h1_errors;// = {0.0};  
//             //  h1_errors.fill(0.0);
//             for (int j = 0; j < h1_size; ++j) {
//                 for (int k = 0; k < h2_size; ++k) {
//                     h1_errors[j] += h2_errors[k] * weight_h1_h2[j][k];
//                 }
//                 h1_errors[j] *= tanh_derivative(hidden1[j]);
//             }

//             // Update weights and biases
//             for (int j = 0; j < out_size; ++j) {
//                 for (int k = 0; k < h3_size; ++k) {
//                     weight_h3_out[k][j] += learning_rate * output_errors[j] * hidden3[k];
//                 }
//                 bias_out[j] += learning_rate * output_errors[j];
//             }

//             for (int j = 0; j < h3_size; ++j) {
//                 for (int k = 0; k < h2_size; ++k) {
//                     weight_h2_h3[k][j] += learning_rate * h3_errors[j] * hidden2[k];
//                 }
//                 bias_h3[j] += learning_rate * h3_errors[j];
//             }

//             for (int j = 0; j < h2_size; ++j) {
//                 for (int k = 0; k < h1_size; ++k) {
//                     weight_h1_h2[k][j] += learning_rate * h2_errors[j] * hidden1[k];
//                 }
//                 bias_h2[j] += learning_rate * h2_errors[j];
//             }

//             for (int j = 0; j < h1_size; ++j) {
//                 for (int k = 0; k < in_size; ++k) {
//                     weight_in_h1[k][j] += learning_rate * h1_errors[j] * saved_inputs[i][k];
//                 }
//                 bias_h1[j] += learning_rate * h1_errors[j];
//             }
//         }

//         std::cout << "Epoch " << epoch + 1 << " - Loss: " << total_loss / saved_inputs.size() << "\n";
//     }
// }

// void MLP_TANH::initialize_weights_and_biases() {
//     srand(static_cast<unsigned>(time(0)));

//     for (int i = 0; i < in_size; ++i) {
//         for (int j = 0; j < h1_size; ++j) {
//             weight_in_h1[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
//         }
//     }

//     for (int i = 0; i < h1_size; ++i) {
//         for (int j = 0; j < h2_size; ++j) {
//             weight_h1_h2[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
//         }
//     }

//     for (int i = 0; i < h2_size; ++i) {
//         for (int j = 0; j < h3_size; ++j) {
//             weight_h2_h3[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
//         }
//     }

//     for (int i = 0; i < h3_size; ++i) {
//         for (int j = 0; j < out_size; ++j) {
//             weight_h3_out[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
//         }
//     }

//     for (int i = 0; i < h1_size; ++i) {
//         bias_h1[i] = ((double)rand() / RAND_MAX) * 2 - 1;
//     }

//     for (int i = 0; i < h2_size; ++i) {
//         bias_h2[i] = ((double)rand() / RAND_MAX) * 2 - 1;
//     }

//     for (int i = 0; i < h3_size; ++i) {
//         bias_h3[i] = ((double)rand() / RAND_MAX) * 2 - 1;
//     }

//     for (int i = 0; i < out_size; ++i) {
//         bias_out[i] = ((double)rand() / RAND_MAX) * 2 - 1;
//     }
// }

// void MLP_TANH::clear_data() {
//     for(int i=0; i < accumulated_inputs.size(); i++) {
//         accumulated_inputs[i].fill(0.0);
//         accumulated_targets[i].fill(0.0);
//         accumulated_data_size = 0;
//     }
//     // accumulated_inputs.clear();
//     // accumulated_targets.clear();
//     std::cout << "Accumulated data size: " << accumulated_data_size <<"\n";
// }

// void MLP_TANH::accumulate_data(const  idsp::c_vector<double, 3>& inputs, const  idsp::c_vector<double, 6>& targets) {
//     // accumulated_inputs.push_back(inputs);
//     int i = 0;
//     for (double val : inputs) {
//         std::cout << "NoOf AccumData: " << accumulated_data_size << " in "<< i << " inputs: " << val << " ";
//         accumulated_inputs[accumulated_data_size][i] = val;
//         i++;
//     }
//     // accumulated_targets.push_back(targets);
//     int o = 0;
//     for (double val : targets) {
//         std::cout << "NoOf AccumData: " << accumulated_data_size << " out " << o << " targets: " << val << " ";
//         accumulated_targets[accumulated_data_size][o] = val;
//         o++;
//     }
//     accumulated_data_size++;
//     std::cout << "Accumulated data size: " << accumulated_data_size <<"\n";
// }

// void MLP_TANH::retrain(double learning_rate, int epochs) {
//     if (!accumulated_inputs.empty() && !accumulated_targets.empty()) {
//         train(accumulated_inputs, accumulated_targets, learning_rate, epochs);
//     }
// }

//  idsp::c_vector<double, 6> MLP_TANH::generate_output() {
//      idsp::c_vector<double, 6> output = forward(this->net_in);

//     // std::cout << "Test Input: [";
//     // for (double val : input) {
//     //     std::cout << val << " ";
//     // }
//     // std::cout << "]\nOutput: [";
//     // for (double val : output) {
//     //     std::cout << val << " ";
//     // }
//     // std::cout << "]\n";


//     // std::cout << "Output: [";
//     // for (double val : output) {
//     //     std::cout << val << " ";
//     // }
//     // std::cout << "]\n";

//     return output;
// }

// // int main() {
// //     MLP_TANH mlp(3, 5, 4, 5, 6);

// //     std::vector<std::vector<double>> inputs = {
// //         {0.5, -0.2, 0.8},
// //         {0.1, 0.4, -0.3},
// //         {-0.7, 0.2, 0.6}
// //     };

// //     std::vector<std::vector<double>> targets = {
// //         {0.6, -0.4, 0.7, -0.3, 0.2, 0.5},
// //         {-0.1, 0.3, -0.6, 0.4, -0.5, 0.8},
// //         {0.2, -0.3, 0.4, -0.5, 0.6, 0.7}
// //     };

// //     mlp.train(inputs, targets, 0.005, 10000);

// //     std::vector<double> test_input = {0.5, -0.2, 0.8};
// //     mlp.generate_output(test_input);

// //     return 0;
// // }
