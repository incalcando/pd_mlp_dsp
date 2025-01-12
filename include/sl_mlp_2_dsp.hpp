#ifndef sl_mlp_2_DSP_HPP
#define sl_mlp_2_DSP_HPP

#include "idsp/buffer_types.hpp"
#include "mlp_tanh.hpp"
#include <array> // Added for std::array

static constexpr size_t MLP_INPUT_SIZE = 3;
static constexpr std::array<size_t,3> MLP_HIDDEN_LAYERS{15, 12, 15};
static constexpr size_t MLP_OUTPUT_SIZE = 6;
static constexpr size_t MLP_TRAINING_EPOCHS = 1000;

using DspBuffer = idsp::PolySampleBufferDynamic<2>;

class Sl_mlp_2
{
    public:
        Sl_mlp_2() : gain(1.0f),
        training_epochs(MLP_TRAINING_EPOCHS),
        mlp(training_epochs)      
        {
        }

        void process(const DspBuffer& input, DspBuffer& output);
        void set_gain(const float f) {gain = f;}
        // void train(const float f) {mlp.train();}
        void set_mlp_x(const float f) {mlp.fill_input_vec(0, f);}
        void set_mlp_y(const float f) {mlp.fill_input_vec(1, f);}
        void set_mlp_z(const float f) {mlp.fill_input_vec(2, f);}
        void set_data_pair(const  idsp::c_vector<double, MLP_INPUT_SIZE>& inputs, const  idsp::c_vector<double, MLP_OUTPUT_SIZE>& targets) {mlp.accumulate_data(inputs, targets);} // Changed to  idsp::c_vector
        void clear_data() {mlp.clear_data();}
        void retrain() {mlp.retrain(0.01, training_epochs);}
        void set_training_epochs(const int epochs) 
        {   
            training_epochs = epochs; 
            mlp.set_training_amount(training_epochs);
        }

        // std::vector<double> generate_output() {
        //     std::vector<double> test_input = {0.1, 0.2, 0.3};
        //     return mlp.generate_output(test_input);
        // }
         idsp::c_vector<double, MLP_OUTPUT_SIZE> generate_output(); // Changed to  idsp::c_vector
        

    private:
        float gain;
        int training_epochs;
        MLP_TANH<MLP_INPUT_SIZE, MLP_HIDDEN_LAYERS[0], MLP_HIDDEN_LAYERS[1], MLP_HIDDEN_LAYERS[2], MLP_OUTPUT_SIZE> mlp;
};

#endif // sl_mlp_2_DSP_HPP
