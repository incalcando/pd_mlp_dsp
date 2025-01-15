//// 2 LAYER MLP

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t OUT_SIZE>
MLP_2LYR<IN_SIZE, H1_SIZE, H2_SIZE, OUT_SIZE>::MLP_2LYR(size_t epochs) : 
    training_epochs(epochs), 
    accumulated_data_size(0) 
{
    initialize_weights_and_biases();
    if(IN_SIZE != 3 || OUT_SIZE != 6) {
        throw std::invalid_argument("IN_SIZE must be 3 and OUT_SIZE must be 6");
    }
    accumulated_inputs.resize(7);
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0., 0., 0.});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{1., 0., 0.});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0., 1., 0.});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0., 0., 1.});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0.5, 0.2, 0.8});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0.1, 0.4, 0.3});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0.7, 0.2, 0.6});

    accumulated_targets.resize(7);
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.9, -0.9, 0.0, 0.0, 0.0, 0.0});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.0, 0.0, 0.9, -0.9, 0.0, 0.0});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.0, 0.0, 0.0, 0.0, 0.9, -0.9});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.6, -0.4, 0.7, -0.3, 0.2, 0.5});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{-0.1, 0.3, -0.6, 0.4, -0.5, 0.8});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.2, -0.3, 0.4, -0.5, 0.6, 0.7});
    
    accumulated_data_size = 7;

    train(accumulated_inputs, accumulated_targets, 0.01, training_epochs);

}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t OUT_SIZE>
idsp::c_vector<double, OUT_SIZE> MLP_2LYR<IN_SIZE, H1_SIZE, H2_SIZE, OUT_SIZE>::forward(const idsp::c_vector<double, IN_SIZE>& current_input, bool linear_output) {
    if (current_input.size() != IN_SIZE) {
        throw std::invalid_argument("Input size does not match");
    }    
    // Hidden layer 1
     idsp::c_vector<double, H1_SIZE> hidden1;// = {0.0}; 
    //  hidden1.fill(0.0); 
    for (int i = 0; i < H1_SIZE; ++i) {
        for (int j = 0; j < IN_SIZE; ++j) {
            hidden1[i] += current_input[j] * weight_in_h1[j][i];
        }
        hidden1[i] += bias_h1[i];
        hidden1[i] = tanh_activation(hidden1[i]);
    }

    // Hidden layer 2
     idsp::c_vector<double, H2_SIZE> hidden2;// = {0.0};  
        // hidden2.fill(0.0);
    for (int i = 0; i < H2_SIZE; ++i) {
        for (int j = 0; j < H1_SIZE; ++j) {
            hidden2[i] += hidden1[j] * weight_h1_h2[j][i];
        }
        hidden2[i] += bias_h2[i];
        hidden2[i] = tanh_activation(hidden2[i]);
    }

    // Output layer
     idsp::c_vector<double, OUT_SIZE> outputs;// = {0.0};  
        // outputs.fill(0.0);
    for (int i = 0; i < OUT_SIZE; ++i) {
        for (int j = 0; j < H2_SIZE; ++j) {
            outputs[i] += hidden2[j] * weight_h2_out[j][i];
        }
        outputs[i] += bias_out[i];
        outputs[i] = linear_output ? outputs[i] : tanh_activation(outputs[i]);
    }

    return outputs;
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t OUT_SIZE>
void MLP_2LYR<IN_SIZE, H1_SIZE, H2_SIZE, OUT_SIZE>::train(
        const idsp::c_vector<idsp::c_vector<double, IN_SIZE>, 99>& saved_inputs, 
        const idsp::c_vector<idsp::c_vector<double, OUT_SIZE>, 99>& targets, 
        double learning_rate, 
        int epochs) 
    {
    if (saved_inputs.size() != targets.size()) {
        throw std::invalid_argument("Inputs and targets must have the same size");
    }
    std::cout << "Training MLP with " << accumulated_data_size << " data sets\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (size_t i = 0; i < saved_inputs.size(); ++i) {
        // for (size_t i = 0; i < saved_inputs.size(); ++i) {
            // Forward pass
             idsp::c_vector<double, H1_SIZE> hidden1;// = {0.0};
            //  hidden1.fill(0.0);
             idsp::c_vector<double, H2_SIZE> hidden2;// = {0.0};
            //  hidden2.fill(0.0);
             idsp::c_vector<double, OUT_SIZE> outputs;// = {0.0};
            //  outputs.fill(0.0);

            // Hidden layer 1
            for (int j = 0; j < H1_SIZE; ++j) {
                for (int k = 0; k < IN_SIZE; ++k) {
                    hidden1[j] += saved_inputs[i][k] * weight_in_h1[k][j];
                }
                hidden1[j] += bias_h1[j];
                hidden1[j] = tanh_activation(hidden1[j]);
            }

            // Hidden layer 2
            for (int j = 0; j < H2_SIZE; ++j) {
                for (int k = 0; k < H1_SIZE; ++k) {
                    hidden2[j] += hidden1[k] * weight_h1_h2[k][j];
                }
                hidden2[j] += bias_h2[j];
                hidden2[j] = tanh_activation(hidden2[j]);
            }

            // Output layer
            for (int j = 0; j < OUT_SIZE; ++j) {
                for (int k = 0; k < H2_SIZE; ++k) {
                    outputs[j] += hidden2[k] * weight_h2_out[k][j];
                }
                outputs[j] += bias_out[j];
                outputs[j] = tanh_activation(outputs[j]);
            }

            // Compute loss
            total_loss += mean_squared_error(outputs, targets[i]);

            // Backpropagation
             idsp::c_vector<double, OUT_SIZE> output_errors;// = {0.0};  
            //  output_errors.fill(0.0);
            for (int j = 0; j < OUT_SIZE; ++j) {
                output_errors[j] = (targets[i][j] - outputs[j]) * tanh_derivative(outputs[j]);
            }

             idsp::c_vector<double, H2_SIZE> h2_errors;// = {0.0};  
            //  h3_errors.fill(0.0);
            for (int j = 0; j < H2_SIZE; ++j) {
                for (int k = 0; k < OUT_SIZE; ++k) {
                    h2_errors[j] += output_errors[k] * weight_h2_out[j][k];
                }
                h2_errors[j] *= tanh_derivative(hidden2[j]);
            }

             idsp::c_vector<double, H1_SIZE> h1_errors;// = {0.0};  
            //  h1_errors.fill(0.0);
            for (int j = 0; j < H1_SIZE; ++j) {
                for (int k = 0; k < H2_SIZE; ++k) {
                    h1_errors[j] += h2_errors[k] * weight_h1_h2[j][k];
                }
                h1_errors[j] *= tanh_derivative(hidden1[j]);
            }

            // Update weights and biases
            for (int j = 0; j < OUT_SIZE; ++j) {
                for (int k = 0; k < H2_SIZE; ++k) {
                    weight_h2_out[k][j] += learning_rate * output_errors[j] * hidden2[k];
                }
                bias_out[j] += learning_rate * output_errors[j];
            }

            for (int j = 0; j < H2_SIZE; ++j) {
                for (int k = 0; k < H1_SIZE; ++k) {
                    weight_h1_h2[k][j] += learning_rate * h2_errors[j] * hidden1[k];
                }
                bias_h2[j] += learning_rate * h2_errors[j];
            }

            for (int j = 0; j < H1_SIZE; ++j) {
                for (int k = 0; k < IN_SIZE; ++k) {
                    weight_in_h1[k][j] += learning_rate * h1_errors[j] * saved_inputs[i][k];
                }
                bias_h1[j] += learning_rate * h1_errors[j];
            }
        }
        if(epoch == epochs - 1)
            std::cout << "Epoch " << epoch + 1 << " - Loss: " << total_loss / saved_inputs.size() << "\n";
    }
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t OUT_SIZE>
idsp::c_vector<double, OUT_SIZE> MLP_2LYR<IN_SIZE, H1_SIZE, H2_SIZE, OUT_SIZE>::generate_output() {
    // Function implementation
    idsp::c_vector<double, OUT_SIZE> output;

    output = forward(net_in, false);

    return output;
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t OUT_SIZE>
void MLP_2LYR<IN_SIZE, H1_SIZE, H2_SIZE, OUT_SIZE>::clear_data() {
    
    for(int i=0; i < accumulated_inputs.size(); i++) {
        accumulated_inputs[i].fill(0.0);
        accumulated_targets[i].fill(0.0);
        accumulated_data_size = 0;
    }
    // accumulated_inputs.clear();
    // accumulated_targets.clear();
    std::cout << "Accumulated data size: " << accumulated_data_size <<"\n";
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t OUT_SIZE>
void MLP_2LYR<IN_SIZE, H1_SIZE, H2_SIZE, OUT_SIZE>::accumulate_data(const idsp::c_vector<double, IN_SIZE>& inputs, const idsp::c_vector<double, OUT_SIZE>& targets) {
    
    // accumulated_inputs.push_back(inputs);
    int i = 0;
    for (double val : inputs) {
        std::cout << "NoOf AccumData: " << accumulated_data_size << " in "<< i << " inputs: " << val << " ";
        accumulated_inputs[accumulated_data_size][i] = val;
        i++;
    }
    // accumulated_targets.push_back(targets);
    int o = 0;
    for (double val : targets) {
        std::cout << "NoOf AccumData: " << accumulated_data_size << " out " << o << " targets: " << val << " ";
        accumulated_targets[accumulated_data_size][o] = val;
        o++;
    }
    accumulated_data_size++;
    std::cout << "Accumulated data size: " << accumulated_data_size <<"\n";
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t OUT_SIZE>
void MLP_2LYR<IN_SIZE, H1_SIZE, H2_SIZE, OUT_SIZE>::retrain(double learning_rate, int epochs) {
    
    if (!accumulated_inputs.empty() && !accumulated_targets.empty()) {
        train(accumulated_inputs, accumulated_targets, learning_rate, epochs);
    }
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t OUT_SIZE>
void MLP_2LYR<IN_SIZE, H1_SIZE, H2_SIZE, OUT_SIZE>::initialize_weights_and_biases() {
    
    srand(static_cast<unsigned>(time(0)));

    for (int i = 0; i < IN_SIZE; ++i) {
        for (int j = 0; j < H1_SIZE; ++j) {
            weight_in_h1[i][j] = get_random_bipolar_normalised();
        }
    }

    for (int i = 0; i < H1_SIZE; ++i) {
        for (int j = 0; j < H2_SIZE; ++j) {
            weight_h1_h2[i][j] = get_random_bipolar_normalised();
        }
    }

    for (int i = 0; i < H2_SIZE; ++i) {
        for (int j = 0; j < OUT_SIZE; ++j) {
            weight_h2_out[i][j] = get_random_bipolar_normalised();
        }
    }

    for (int i = 0; i < H1_SIZE; ++i) {
        bias_h1[i] = get_random_bipolar_normalised();
    }

    for (int i = 0; i < H2_SIZE; ++i) {
        bias_h2[i] = get_random_bipolar_normalised();
    }

    for (int i = 0; i < OUT_SIZE; ++i) {
        bias_out[i] = get_random_bipolar_normalised();
    }
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t OUT_SIZE>
idsp::c_vector<double, IN_SIZE> MLP_2LYR<IN_SIZE, H1_SIZE, H2_SIZE, OUT_SIZE>::find_closest_data_set(const idsp::c_vector<double, OUT_SIZE>& target) {
    double min_distance = std::numeric_limits<double>::max();
    size_t closest_index = 0;

    for (size_t i = 0; i < accumulated_targets.size(); ++i) {
        double distance = 0.0;
        for (size_t j = 0; j < OUT_SIZE; ++j) {
            distance += std::pow(accumulated_targets[i][j] - target[j], 2);
        }
        distance = std::sqrt(distance);

        if (distance < min_distance) {
            min_distance = distance;
            closest_index = i;
        }
    }

    return accumulated_inputs[closest_index];
}

//// 3 LAYER MLP

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t OUT_SIZE>
MLP_3LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, OUT_SIZE>::MLP_3LYR(size_t epochs) : 
    training_epochs(epochs), 
    accumulated_data_size(0) 
{
    initialize_weights_and_biases();
    if(IN_SIZE != 3 || OUT_SIZE != 6) {
        throw std::invalid_argument("IN_SIZE must be 3 and OUT_SIZE must be 6");
    }
    accumulated_inputs.resize(7);
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0., 0., 0.});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{1., 0., 0.});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0., 1., 0.});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0., 0., 1.});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0.5, 0.2, 0.8});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0.1, 0.4, 0.3});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0.7, 0.2, 0.6});

    accumulated_targets.resize(7);
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.9, -0.9, 0.0, 0.0, 0.0, 0.0});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.0, 0.0, 0.9, -0.9, 0.0, 0.0});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.0, 0.0, 0.0, 0.0, 0.9, -0.9});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.6, -0.4, 0.7, -0.3, 0.2, 0.5});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{-0.1, 0.3, -0.6, 0.4, -0.5, 0.8});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.2, -0.3, 0.4, -0.5, 0.6, 0.7});
    
    accumulated_data_size = 7;

    train(accumulated_inputs, accumulated_targets, 0.01, training_epochs);

}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t OUT_SIZE>
idsp::c_vector<double, OUT_SIZE> MLP_3LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, OUT_SIZE>::forward(const idsp::c_vector<double, IN_SIZE>& current_input, bool linear_output) {
    if (current_input.size() != IN_SIZE) {
        throw std::invalid_argument("Input size does not match");
    }    
    // Hidden layer 1
     idsp::c_vector<double, H1_SIZE> hidden1;// = {0.0}; 
    //  hidden1.fill(0.0); 
    for (int i = 0; i < H1_SIZE; ++i) {
        for (int j = 0; j < IN_SIZE; ++j) {
            hidden1[i] += current_input[j] * weight_in_h1[j][i];
        }
        hidden1[i] += bias_h1[i];
        hidden1[i] = tanh_activation(hidden1[i]);
    }

    // Hidden layer 2
     idsp::c_vector<double, H2_SIZE> hidden2;// = {0.0};  
        // hidden2.fill(0.0);
    for (int i = 0; i < H2_SIZE; ++i) {
        for (int j = 0; j < H1_SIZE; ++j) {
            hidden2[i] += hidden1[j] * weight_h1_h2[j][i];
        }
        hidden2[i] += bias_h2[i];
        hidden2[i] = tanh_activation(hidden2[i]);
    }

    // Hidden layer 3
     idsp::c_vector<double, H3_SIZE> hidden3;// = {0.0};  
    //  hidden3.fill(0.0);
    for (int i = 0; i < H3_SIZE; ++i) {
        for (int j = 0; j < H2_SIZE; ++j) {
            hidden3[i] += hidden2[j] * weight_h2_h3[j][i];
        }
        hidden3[i] += bias_h3[i];
        hidden3[i] = tanh_activation(hidden3[i]);
    }

    // Output layer
     idsp::c_vector<double, OUT_SIZE> outputs;// = {0.0};  
        // outputs.fill(0.0);
    for (int i = 0; i < OUT_SIZE; ++i) {
        for (int j = 0; j < H3_SIZE; ++j) {
            outputs[i] += hidden3[j] * weight_h3_out[j][i];
        }
        outputs[i] += bias_out[i];
        outputs[i] = linear_output ? outputs[i] : tanh_activation(outputs[i]);
    }

    return outputs;
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t OUT_SIZE>
void MLP_3LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, OUT_SIZE>::train(
        const idsp::c_vector<idsp::c_vector<double, IN_SIZE>, 99>& saved_inputs, 
        const idsp::c_vector<idsp::c_vector<double, OUT_SIZE>, 99>& targets, 
        double learning_rate, 
        int epochs) 
    {
    if (saved_inputs.size() != targets.size()) {
        throw std::invalid_argument("Inputs and targets must have the same size");
    }
    std::cout << "Training MLP with " << accumulated_data_size << " data sets\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (size_t i = 0; i < saved_inputs.size(); ++i) {
        // for (size_t i = 0; i < saved_inputs.size(); ++i) {
            // Forward pass
             idsp::c_vector<double, H1_SIZE> hidden1;// = {0.0};
            //  hidden1.fill(0.0);
             idsp::c_vector<double, H2_SIZE> hidden2;// = {0.0};
            //  hidden2.fill(0.0);
             idsp::c_vector<double, H3_SIZE> hidden3;// = {0.0};
            //  hidden3.fill(0.0);
             idsp::c_vector<double, OUT_SIZE> outputs;// = {0.0};
            //  outputs.fill(0.0);

            // Hidden layer 1
            for (int j = 0; j < H1_SIZE; ++j) {
                for (int k = 0; k < IN_SIZE; ++k) {
                    hidden1[j] += saved_inputs[i][k] * weight_in_h1[k][j];
                }
                hidden1[j] += bias_h1[j];
                hidden1[j] = tanh_activation(hidden1[j]);
            }

            // Hidden layer 2
            for (int j = 0; j < H2_SIZE; ++j) {
                for (int k = 0; k < H1_SIZE; ++k) {
                    hidden2[j] += hidden1[k] * weight_h1_h2[k][j];
                }
                hidden2[j] += bias_h2[j];
                hidden2[j] = tanh_activation(hidden2[j]);
            }

            // Hidden layer 3
            for (int j = 0; j < H3_SIZE; ++j) {
                for (int k = 0; k < H2_SIZE; ++k) {
                    hidden3[j] += hidden2[k] * weight_h2_h3[k][j];
                }
                hidden3[j] += bias_h3[j];
                hidden3[j] = tanh_activation(hidden3[j]);
            }

            // Output layer
            for (int j = 0; j < OUT_SIZE; ++j) {
                for (int k = 0; k < H3_SIZE; ++k) {
                    outputs[j] += hidden3[k] * weight_h3_out[k][j];
                }
                outputs[j] += bias_out[j];
                outputs[j] = tanh_activation(outputs[j]);
            }

            // Compute loss
            total_loss += mean_squared_error(outputs, targets[i]);

            // Backpropagation
             idsp::c_vector<double, OUT_SIZE> output_errors;// = {0.0};  
            //  output_errors.fill(0.0);
            for (int j = 0; j < OUT_SIZE; ++j) {
                output_errors[j] = (targets[i][j] - outputs[j]) * tanh_derivative(outputs[j]);
            }

             idsp::c_vector<double, H3_SIZE> h3_errors;// = {0.0};  
            //  h3_errors.fill(0.0);
            for (int j = 0; j < H3_SIZE; ++j) {
                for (int k = 0; k < OUT_SIZE; ++k) {
                    h3_errors[j] += output_errors[k] * weight_h3_out[j][k];
                }
                h3_errors[j] *= tanh_derivative(hidden3[j]);
            }

             idsp::c_vector<double, H2_SIZE> h2_errors;// = {0.0};  
                // h2_errors.fill(0.0);
            for (int j = 0; j < H2_SIZE; ++j) {
                for (int k = 0; k < H3_SIZE; ++k) {
                    h2_errors[j] += h3_errors[k] * weight_h2_h3[j][k];
                }
                h2_errors[j] *= tanh_derivative(hidden2[j]);
            }

             idsp::c_vector<double, H1_SIZE> h1_errors;// = {0.0};  
            //  h1_errors.fill(0.0);
            for (int j = 0; j < H1_SIZE; ++j) {
                for (int k = 0; k < H2_SIZE; ++k) {
                    h1_errors[j] += h2_errors[k] * weight_h1_h2[j][k];
                }
                h1_errors[j] *= tanh_derivative(hidden1[j]);
            }

            // Update weights and biases
            for (int j = 0; j < OUT_SIZE; ++j) {
                for (int k = 0; k < H3_SIZE; ++k) {
                    weight_h3_out[k][j] += learning_rate * output_errors[j] * hidden3[k];
                }
                bias_out[j] += learning_rate * output_errors[j];
            }

            for (int j = 0; j < H3_SIZE; ++j) {
                for (int k = 0; k < H2_SIZE; ++k) {
                    weight_h2_h3[k][j] += learning_rate * h3_errors[j] * hidden2[k];
                }
                bias_h3[j] += learning_rate * h3_errors[j];
            }

            for (int j = 0; j < H2_SIZE; ++j) {
                for (int k = 0; k < H1_SIZE; ++k) {
                    weight_h1_h2[k][j] += learning_rate * h2_errors[j] * hidden1[k];
                }
                bias_h2[j] += learning_rate * h2_errors[j];
            }

            for (int j = 0; j < H1_SIZE; ++j) {
                for (int k = 0; k < IN_SIZE; ++k) {
                    weight_in_h1[k][j] += learning_rate * h1_errors[j] * saved_inputs[i][k];
                }
                bias_h1[j] += learning_rate * h1_errors[j];
            }
        }
        if(epoch == epochs - 1)
            std::cout << "Epoch " << epoch + 1 << " - Loss: " << total_loss / saved_inputs.size() << "\n";
    }
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t OUT_SIZE>
idsp::c_vector<double, OUT_SIZE> MLP_3LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, OUT_SIZE>::generate_output() {
    // Function implementation
    idsp::c_vector<double, OUT_SIZE> output;

    output = forward(net_in, false);

    return output;
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t OUT_SIZE>
void MLP_3LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, OUT_SIZE>::clear_data() {
    
    for(int i=0; i < accumulated_inputs.size(); i++) {
        accumulated_inputs[i].fill(0.0);
        accumulated_targets[i].fill(0.0);
        accumulated_data_size = 0;
    }
    // accumulated_inputs.clear();
    // accumulated_targets.clear();
    std::cout << "Accumulated data size: " << accumulated_data_size <<"\n";
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t OUT_SIZE>
void MLP_3LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, OUT_SIZE>::accumulate_data(const idsp::c_vector<double, IN_SIZE>& inputs, const idsp::c_vector<double, OUT_SIZE>& targets) {
    
    // accumulated_inputs.push_back(inputs);
    int i = 0;
    for (double val : inputs) {
        std::cout << "NoOf AccumData: " << accumulated_data_size << " in "<< i << " inputs: " << val << " ";
        accumulated_inputs[accumulated_data_size][i] = val;
        i++;
    }
    // accumulated_targets.push_back(targets);
    int o = 0;
    for (double val : targets) {
        std::cout << "NoOf AccumData: " << accumulated_data_size << " out " << o << " targets: " << val << " ";
        accumulated_targets[accumulated_data_size][o] = val;
        o++;
    }
    accumulated_data_size++;
    std::cout << "Accumulated data size: " << accumulated_data_size <<"\n";
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t OUT_SIZE>
void MLP_3LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, OUT_SIZE>::retrain(double learning_rate, int epochs) {
    
    if (!accumulated_inputs.empty() && !accumulated_targets.empty()) {
        train(accumulated_inputs, accumulated_targets, learning_rate, epochs);
    }
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t OUT_SIZE>
void MLP_3LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, OUT_SIZE>::initialize_weights_and_biases() {
    
    srand(static_cast<unsigned>(time(0)));

    for (int i = 0; i < IN_SIZE; ++i) {
        for (int j = 0; j < H1_SIZE; ++j) {
            weight_in_h1[i][j] = get_random_bipolar_normalised();
        }
    }

    for (int i = 0; i < H1_SIZE; ++i) {
        for (int j = 0; j < H2_SIZE; ++j) {
            weight_h1_h2[i][j] = get_random_bipolar_normalised();
        }
    }

    for (int i = 0; i < H2_SIZE; ++i) {
        for (int j = 0; j < H3_SIZE; ++j) {
            weight_h2_h3[i][j] = get_random_bipolar_normalised();
        }
    }

    for (int i = 0; i < H3_SIZE; ++i) {
        for (int j = 0; j < OUT_SIZE; ++j) {
            weight_h3_out[i][j] = get_random_bipolar_normalised();
        }
    }

    for (int i = 0; i < H1_SIZE; ++i) {
        bias_h1[i] = get_random_bipolar_normalised();
    }

    for (int i = 0; i < H2_SIZE; ++i) {
        bias_h2[i] = get_random_bipolar_normalised();
    }

    for (int i = 0; i < H3_SIZE; ++i) {
        bias_h3[i] = get_random_bipolar_normalised();
    }

    for (int i = 0; i < OUT_SIZE; ++i) {
        bias_out[i] = get_random_bipolar_normalised();
    }
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t OUT_SIZE>
idsp::c_vector<double, IN_SIZE> MLP_3LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, OUT_SIZE>::find_closest_data_set(const idsp::c_vector<double, OUT_SIZE>& target) {
    double min_distance = std::numeric_limits<double>::max();
    size_t closest_index = 0;

    for (size_t i = 0; i < accumulated_targets.size(); ++i) {
        double distance = 0.0;
        for (size_t j = 0; j < OUT_SIZE; ++j) {
            distance += std::pow(accumulated_targets[i][j] - target[j], 2);
        }
        distance = std::sqrt(distance);

        if (distance < min_distance) {
            min_distance = distance;
            closest_index = i;
        }
    }

    return accumulated_inputs[closest_index];
}


//// 4 LAYER MLP

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t H4_SIZE, size_t OUT_SIZE>
MLP_4LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, H4_SIZE, OUT_SIZE>::MLP_4LYR(size_t epochs) : 
    training_epochs(epochs), 
    accumulated_data_size(0) 
{
    initialize_weights_and_biases();
    if(IN_SIZE != 3 || OUT_SIZE != 6) {
        throw std::invalid_argument("IN_SIZE must be 3 and OUT_SIZE must be 6");
    }
    accumulated_inputs.resize(7);
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0., 0., 0.});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{1., 0., 0.});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0., 1., 0.});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0., 0., 1.});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0.5, 0.2, 0.8});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0.1, 0.4, 0.3});
    accumulated_inputs.push_back(idsp::c_vector<double, 3>{0.7, 0.2, 0.6});

    accumulated_targets.resize(7);
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.9, -0.9, 0.0, 0.0, 0.0, 0.0});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.0, 0.0, 0.9, -0.9, 0.0, 0.0});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.0, 0.0, 0.0, 0.0, 0.9, -0.9});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.6, -0.4, 0.7, -0.3, 0.2, 0.5});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{-0.1, 0.3, -0.6, 0.4, -0.5, 0.8});
    accumulated_targets.push_back(idsp::c_vector<double, 6>{0.2, -0.3, 0.4, -0.5, 0.6, 0.7});
    
    accumulated_data_size = 7;

    train(accumulated_inputs, accumulated_targets, 0.01, training_epochs);

}

//// forward() function for 4 layer MLP

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t H4_SIZE, size_t OUT_SIZE>
idsp::c_vector<double, OUT_SIZE> MLP_4LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, H4_SIZE, OUT_SIZE>::forward(const idsp::c_vector<double, IN_SIZE>& current_input, bool linear_output) {
    if (current_input.size() != IN_SIZE) {
        throw std::invalid_argument("Input size does not match");
    }    
    // Hidden layer 1
     idsp::c_vector<double, H1_SIZE> hidden1;// = {0.0}; 
    //  hidden1.fill(0.0); 
    for (int i = 0; i < H1_SIZE; ++i) {
        for (int j = 0; j < IN_SIZE; ++j) {
            hidden1[i] += current_input[j] * weight_in_h1[j][i];
        }
        hidden1[i] += bias_h1[i];
        hidden1[i] = tanh_activation(hidden1[i]);
    }

    // Hidden layer 2
     idsp::c_vector<double, H2_SIZE> hidden2;// = {0.0};  
        // hidden2.fill(0.0);
    for (int i = 0; i < H2_SIZE; ++i) {
        for (int j = 0; j < H1_SIZE; ++j) {
            hidden2[i] += hidden1[j] * weight_h1_h2[j][i];
        }
        hidden2[i] += bias_h2[i];
        hidden2[i] = tanh_activation(hidden2[i]);
    }

    // Hidden layer 3
     idsp::c_vector<double, H3_SIZE> hidden3;// = {0.0};  
    //  hidden3.fill(0.0);
    for (int i = 0; i < H3_SIZE; ++i) {
        for (int j = 0; j < H2_SIZE; ++j) {
            hidden3[i] += hidden2[j] * weight_h2_h3[j][i];
        }
        hidden3[i] += bias_h3[i];
        hidden3[i] = tanh_activation(hidden3[i]);
    }

    // Hidden layer 4
     idsp::c_vector<double, H4_SIZE> hidden4;// = {0.0};  
    //  hidden3.fill(0.0);
    for (int i = 0; i < H4_SIZE; ++i) {
        for (int j = 0; j < H3_SIZE; ++j) {
            hidden4[i] += hidden3[j] * weight_h3_h4[j][i];
        }
        hidden4[i] += bias_h4[i];
        hidden4[i] = tanh_activation(hidden4[i]);
    }

    // Output layer
     idsp::c_vector<double, OUT_SIZE> outputs;// = {0.0};  
        // outputs.fill(0.0);
    for (int i = 0; i < OUT_SIZE; ++i) {
        for (int j = 0; j < H4_SIZE; ++j) {
            outputs[i] += hidden4[j] * weight_h4_out[j][i];
        }
        outputs[i] += bias_out[i];
        outputs[i] = linear_output ? outputs[i] : tanh_activation(outputs[i]);
    }

    return outputs;
}

//// train() function for 4 layer MLP

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t H4_SIZE, size_t OUT_SIZE>
void MLP_4LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, H4_SIZE, OUT_SIZE>::train(
        const idsp::c_vector<idsp::c_vector<double, IN_SIZE>, 99>& saved_inputs, 
        const idsp::c_vector<idsp::c_vector<double, OUT_SIZE>, 99>& targets, 
        double learning_rate, 
        int epochs) 
    {
    if (saved_inputs.size() != targets.size()) {
        throw std::invalid_argument("Inputs and targets must have the same size");
    }
    std::cout << "Training MLP with " << accumulated_data_size << " data sets\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (size_t i = 0; i < saved_inputs.size(); ++i) {
        // for (size_t i = 0; i < saved_inputs.size(); ++i) {
            // Forward pass
             idsp::c_vector<double, H1_SIZE> hidden1;// = {0.0};
             idsp::c_vector<double, H2_SIZE> hidden2;// = {0.0};
             idsp::c_vector<double, H3_SIZE> hidden3;// = {0.0};
             idsp::c_vector<double, H4_SIZE> hidden4;// = {0.0};
             idsp::c_vector<double, OUT_SIZE> outputs;// = {0.0};

            // Hidden layer 1
            for (int j = 0; j < H1_SIZE; ++j) {
                for (int k = 0; k < IN_SIZE; ++k) {
                    hidden1[j] += saved_inputs[i][k] * weight_in_h1[k][j];
                }
                hidden1[j] += bias_h1[j];
                hidden1[j] = tanh_activation(hidden1[j]);
            }

            // Hidden layer 2
            for (int j = 0; j < H2_SIZE; ++j) {
                for (int k = 0; k < H1_SIZE; ++k) {
                    hidden2[j] += hidden1[k] * weight_h1_h2[k][j];
                }
                hidden2[j] += bias_h2[j];
                hidden2[j] = tanh_activation(hidden2[j]);
            }

            // Hidden layer 3
            for (int j = 0; j < H3_SIZE; ++j) {
                for (int k = 0; k < H2_SIZE; ++k) {
                    hidden3[j] += hidden2[k] * weight_h2_h3[k][j];
                }
                hidden3[j] += bias_h3[j];
                hidden3[j] = tanh_activation(hidden3[j]);
            }

            // Hidden layer 4
            for (int j = 0; j < H4_SIZE; ++j) {
                for (int k = 0; k < H3_SIZE; ++k) {
                    hidden4[j] += hidden3[k] * weight_h3_h4[k][j];
                }
                hidden4[j] += bias_h4[j];
                hidden4[j] = tanh_activation(hidden4[j]);
            }

            // Output layer
            for (int j = 0; j < OUT_SIZE; ++j) {
                for (int k = 0; k < H4_SIZE; ++k) {
                    outputs[j] += hidden4[k] * weight_h4_out[k][j];
                }
                outputs[j] += bias_out[j];
                outputs[j] = tanh_activation(outputs[j]);
            }

            // Compute loss
            total_loss += mean_squared_error(outputs, targets[i]);

            // Backpropagation
             idsp::c_vector<double, OUT_SIZE> output_errors;// = {0.0};  
            //  output_errors.fill(0.0);
            for (int j = 0; j < OUT_SIZE; ++j) {
                output_errors[j] = (targets[i][j] - outputs[j]) * tanh_derivative(outputs[j]);
            }

            idsp::c_vector<double, H4_SIZE> h4_errors;// = {0.0};  
            //  h3_errors.fill(0.0);
            for (int j = 0; j < H4_SIZE; ++j) {
                for (int k = 0; k < OUT_SIZE; ++k) {
                    h4_errors[j] += output_errors[k] * weight_h4_out[j][k];
                }
                h4_errors[j] *= tanh_derivative(hidden4[j]);
            }

             idsp::c_vector<double, H3_SIZE> h3_errors;// = {0.0};  
            //  h3_errors.fill(0.0);
            for (int j = 0; j < H3_SIZE; ++j) {
                for (int k = 0; k < H4_SIZE; ++k) {
                    h3_errors[j] += h4_errors[k] * weight_h3_h4[j][k];
                }
                h3_errors[j] *= tanh_derivative(hidden3[j]);
            }

             idsp::c_vector<double, H2_SIZE> h2_errors;// = {0.0};  
                // h2_errors.fill(0.0);
            for (int j = 0; j < H2_SIZE; ++j) {
                for (int k = 0; k < H3_SIZE; ++k) {
                    h2_errors[j] += h3_errors[k] * weight_h2_h3[j][k];
                }
                h2_errors[j] *= tanh_derivative(hidden2[j]);
            }

             idsp::c_vector<double, H1_SIZE> h1_errors;// = {0.0};  
            //  h1_errors.fill(0.0);
            for (int j = 0; j < H1_SIZE; ++j) {
                for (int k = 0; k < H2_SIZE; ++k) {
                    h1_errors[j] += h2_errors[k] * weight_h1_h2[j][k];
                }
                h1_errors[j] *= tanh_derivative(hidden1[j]);
            }

            // Update weights and biases
            for (int j = 0; j < OUT_SIZE; ++j) {
                for (int k = 0; k < H4_SIZE; ++k) {
                    weight_h4_out[k][j] += learning_rate * output_errors[j] * hidden4[k];
                }
                bias_out[j] += learning_rate * output_errors[j];
            }

            for (int j = 0; j < H4_SIZE; ++j) {
                for (int k = 0; k < H3_SIZE; ++k) {
                    weight_h3_h4[k][j] += learning_rate * h4_errors[j] * hidden3[k];
                }
                bias_h4[j] += learning_rate * h4_errors[j];
            }

            for (int j = 0; j < H3_SIZE; ++j) {
                for (int k = 0; k < H2_SIZE; ++k) {
                    weight_h2_h3[k][j] += learning_rate * h3_errors[j] * hidden2[k];
                }
                bias_h3[j] += learning_rate * h3_errors[j];
            }

            for (int j = 0; j < H2_SIZE; ++j) {
                for (int k = 0; k < H1_SIZE; ++k) {
                    weight_h1_h2[k][j] += learning_rate * h2_errors[j] * hidden1[k];
                }
                bias_h2[j] += learning_rate * h2_errors[j];
            }

            for (int j = 0; j < H1_SIZE; ++j) {
                for (int k = 0; k < IN_SIZE; ++k) {
                    weight_in_h1[k][j] += learning_rate * h1_errors[j] * saved_inputs[i][k];
                }
                bias_h1[j] += learning_rate * h1_errors[j];
            }
        }
        if(epoch == epochs - 1)
            std::cout << "Epoch " << epoch + 1 << " - Loss: " << total_loss / saved_inputs.size() << "\n";
    }
}

//// generate_output() function

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t H4_SIZE, size_t OUT_SIZE>
idsp::c_vector<double, OUT_SIZE> MLP_4LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, H4_SIZE, OUT_SIZE>::generate_output() {
    // Function implementation
    idsp::c_vector<double, OUT_SIZE> output;

    output = forward(net_in, false);

    return output;
}

//// clear_data() function

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t H4_SIZE, size_t OUT_SIZE>
void MLP_4LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, H4_SIZE, OUT_SIZE>::clear_data() {
    
    for(int i=0; i < accumulated_inputs.size(); i++) {
        accumulated_inputs[i].fill(0.0);
        accumulated_targets[i].fill(0.0);
        accumulated_data_size = 0;
    }
    // accumulated_inputs.clear();
    // accumulated_targets.clear();
    std::cout << "Accumulated data size: " << accumulated_data_size <<"\n";
}

//// accumulate_data() function

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t H4_SIZE, size_t OUT_SIZE>
void MLP_4LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, H4_SIZE, OUT_SIZE>::accumulate_data(const idsp::c_vector<double, IN_SIZE>& inputs, const idsp::c_vector<double, OUT_SIZE>& targets) {
    
    // accumulated_inputs.push_back(inputs);
    int i = 0;
    for (double val : inputs) {
        std::cout << "No: " << accumulated_data_size << " in "<< i << " inputs: " << val << " ";
        accumulated_inputs[accumulated_data_size][i] = val;
        i++;
    }
    // accumulated_targets.push_back(targets);
    int o = 0;
    for (double val : targets) {
        std::cout << "No: " << accumulated_data_size << " out " << o << " targets: " << val << " ";
        accumulated_targets[accumulated_data_size][o] = val;
        o++;
    }
    accumulated_data_size++;
    std::cout << "Accumulated data size: " << accumulated_data_size <<"\n";
}

//// retrain() function

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t H4_SIZE, size_t OUT_SIZE>
void MLP_4LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, H4_SIZE, OUT_SIZE>::retrain(double learning_rate, int epochs) {
    
    if (!accumulated_inputs.empty() && !accumulated_targets.empty()) {
        train(accumulated_inputs, accumulated_targets, learning_rate, epochs);
    }
}

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t H4_SIZE, size_t OUT_SIZE>
void MLP_4LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, H4_SIZE, OUT_SIZE>::initialize_weights_and_biases() {
    
    srand(static_cast<unsigned>(time(0)));

    for (int i = 0; i < IN_SIZE; ++i) {
        for (int j = 0; j < H1_SIZE; ++j) {
            weight_in_h1[i][j] = get_random_bipolar_normalised();
        }
    }

    for (int i = 0; i < H1_SIZE; ++i) {
        for (int j = 0; j < H2_SIZE; ++j) {
            weight_h1_h2[i][j] = get_random_bipolar_normalised();
        }
    }

    for (int i = 0; i < H2_SIZE; ++i) {
        for (int j = 0; j < H3_SIZE; ++j) {
            weight_h2_h3[i][j] = get_random_bipolar_normalised();
        }
    }

    for (int i = 0; i < H3_SIZE; ++i) {
        for (int j = 0; j < H4_SIZE; ++j) {
            weight_h3_h4[i][j] = get_random_bipolar_normalised();
        }
    }

    for (int i = 0; i < H4_SIZE; ++i) {
        for (int j = 0; j < OUT_SIZE; ++j) {
            weight_h4_out[i][j] = get_random_bipolar_normalised();
        }
    }

    for (int i = 0; i < H1_SIZE; ++i) {
        bias_h1[i] = get_random_bipolar_normalised();
    }

    for (int i = 0; i < H2_SIZE; ++i) {
        bias_h2[i] = get_random_bipolar_normalised();
    }

    for (int i = 0; i < H3_SIZE; ++i) {
        bias_h3[i] = get_random_bipolar_normalised();
    }

    for (int i = 0; i < H4_SIZE; ++i) {
        bias_h4[i] = get_random_bipolar_normalised();
    }

    for (int i = 0; i < OUT_SIZE; ++i) {
        bias_out[i] = get_random_bipolar_normalised();
    }
}

//// find_closest_data_set() function

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t H4_SIZE, size_t OUT_SIZE>
idsp::c_vector<double, IN_SIZE> MLP_4LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, H4_SIZE, OUT_SIZE>::find_closest_data_set(const idsp::c_vector<double, OUT_SIZE>& target) {
    double min_distance = std::numeric_limits<double>::max();
    size_t closest_index = 0;

    for (size_t i = 0; i < accumulated_targets.size(); ++i) {
        double distance = 0.0;
        for (size_t j = 0; j < OUT_SIZE; ++j) {
            distance += std::pow(accumulated_targets[i][j] - target[j], 2);
        }
        distance = std::sqrt(distance);

        if (distance < min_distance) {
            min_distance = distance;
            closest_index = i;
        }
    }

    return accumulated_inputs[closest_index];
}

//// update() function

template <size_t IN_SIZE, size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, size_t H4_SIZE, size_t OUT_SIZE>
void MLP_4LYR<IN_SIZE, H1_SIZE, H2_SIZE, H3_SIZE, H4_SIZE, OUT_SIZE>::update(
    const idsp::c_vector<double, IN_SIZE>& input, 
    const idsp::c_vector<double, OUT_SIZE>& target, 
    double learning_rate) 
{
    // Forward pass
    idsp::c_vector<double, H1_SIZE> hidden1;
    idsp::c_vector<double, H2_SIZE> hidden2;
    idsp::c_vector<double, H3_SIZE> hidden3;
    idsp::c_vector<double, H4_SIZE> hidden4;
    idsp::c_vector<double, OUT_SIZE> outputs;

    // Hidden layer 1
    for (int j = 0; j < H1_SIZE; ++j) {
        for (int k = 0; k < IN_SIZE; ++k) {
            hidden1[j] += input[k] * weight_in_h1[k][j];
        }
        hidden1[j] += bias_h1[j];
        hidden1[j] = tanh_activation(hidden1[j]);
    }

    // Hidden layer 2
    for (int j = 0; j < H2_SIZE; ++j) {
        for (int k = 0; k < H1_SIZE; ++k) {
            hidden2[j] += hidden1[k] * weight_h1_h2[k][j];
        }
        hidden2[j] += bias_h2[j];
        hidden2[j] = tanh_activation(hidden2[j]);
    }

    // Hidden layer 3
    for (int j = 0; j < H3_SIZE; ++j) {
        for (int k = 0; k < H2_SIZE; ++k) {
            hidden3[j] += hidden2[k] * weight_h2_h3[k][j];
        }
        hidden3[j] += bias_h3[j];
        hidden3[j] = tanh_activation(hidden3[j]);
    }

    // Hidden layer 4
    for (int j = 0; j < H4_SIZE; ++j) {
        for (int k = 0; k < H3_SIZE; ++k) {
            hidden4[j] += hidden3[k] * weight_h3_h4[k][j];
        }
        hidden4[j] += bias_h4[j];
        hidden4[j] = tanh_activation(hidden4[j]);
    }

    // Output layer
    for (int j = 0; j < OUT_SIZE; ++j) {
        for (int k = 0; k < H4_SIZE; ++k) {
            outputs[j] += hidden4[k] * weight_h4_out[k][j];
        }
        outputs[j] += bias_out[j];
        outputs[j] = tanh_activation(outputs[j]);
    }

    // Compute loss
    double total_loss = mean_squared_error(outputs, target);
    std::cout << "Update - Loss: " << total_loss << "\n";

    // Print weights and biases before update
    std::cout << "Weights and biases before update:\n";
    for (int i = 0; i < H1_SIZE; ++i) {
        std::cout << "weight_in_h1[" << i << "]: " << weight_in_h1[0][i] << "\n";
    }
    std::cout << "bias_h1[0]: " << bias_h1[0] << "\n";

    // Backpropagation
    idsp::c_vector<double, OUT_SIZE> output_errors;
    for (int j = 0; j < OUT_SIZE; ++j) {
        output_errors[j] = (target[j] - outputs[j]) * tanh_derivative(outputs[j]);
    }

    idsp::c_vector<double, H4_SIZE> h4_errors;
    for (int j = 0; j < H4_SIZE; ++j) {
        for (int k = 0; k < OUT_SIZE; ++k) {
            h4_errors[j] += output_errors[k] * weight_h4_out[j][k];
        }
        h4_errors[j] *= tanh_derivative(hidden4[j]);
    }

    idsp::c_vector<double, H3_SIZE> h3_errors;
    for (int j = 0; j < H3_SIZE; ++j) {
        for (int k = 0; k < H4_SIZE; ++k) {
            h3_errors[j] += h4_errors[k] * weight_h3_h4[j][k];
        }
        h3_errors[j] *= tanh_derivative(hidden3[j]);
    }

    idsp::c_vector<double, H2_SIZE> h2_errors;
    for (int j = 0; j < H2_SIZE; ++j) {
        for (int k = 0; k < H3_SIZE; ++k) {
            h2_errors[j] += h3_errors[k] * weight_h2_h3[j][k];
        }
        h2_errors[j] *= tanh_derivative(hidden2[j]);
    }

    idsp::c_vector<double, H1_SIZE> h1_errors;
    for (int j = 0; j < H1_SIZE; ++j) {
        for (int k = 0; k < H2_SIZE; ++k) {
            h1_errors[j] += h2_errors[k] * weight_h1_h2[j][k];
        }
        h1_errors[j] *= tanh_derivative(hidden1[j]);
    }

    // Update weights and biases
    for (int j = 0; j < OUT_SIZE; ++j) {
        for (int k = 0; k < H4_SIZE; ++k) {
            weight_h4_out[k][j] += learning_rate * output_errors[j] * hidden4[k];
        }
        bias_out[j] += learning_rate * output_errors[j];
    }

    for (int j = 0; j < H4_SIZE; ++j) {
        for (int k = 0; k < H3_SIZE; ++k) {
            weight_h3_h4[k][j] += learning_rate * h4_errors[j] * hidden3[k];
        }
        bias_h4[j] += learning_rate * h4_errors[j];
    }

    for (int j = 0; j < H3_SIZE; ++j) {
        for (int k = 0; k < H2_SIZE; ++k) {
            weight_h2_h3[k][j] += learning_rate * h3_errors[j] * hidden2[k];
        }
        bias_h3[j] += learning_rate * h3_errors[j];
    }

    for (int j = 0; j < H2_SIZE; ++j) {
        for (int k = 0; k < H1_SIZE; ++k) {
            weight_h1_h2[k][j] += learning_rate * h2_errors[j] * hidden1[k];
        }
        bias_h2[j] += learning_rate * h2_errors[j];
    }

    for (int j = 0; j < H1_SIZE; ++j) {
        for (int k = 0; k < IN_SIZE; ++k) {
            weight_in_h1[k][j] += learning_rate * h1_errors[j] * input[k];
        }
        bias_h1[j] += learning_rate * h1_errors[j];
    }

    // Print weights and biases after update
    std::cout << "Weights and biases after update:\n";
    for (int i = 0; i < H1_SIZE; ++i) {
        std::cout << "weight_in_h1[" << i << "]: " << weight_in_h1[0][i] << "\n";
    }
    std::cout << "bias_h1[0]: " << bias_h1[0] << "\n";
}


