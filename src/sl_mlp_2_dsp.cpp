#include "sl_mlp_2_dsp.hpp"
#include "idsp/functions.hpp"

void Sl_mlp_2::process(const DspBuffer& input, DspBuffer& output)
{
    // DSP processing code here
    for(size_t i = 0; i < input[0].size(); i++)
    {
        output[0][i] = input[0][i] * gain;
        output[1][i] = input[1][i] * gain;
    }
}

 idsp::c_vector<double, MLP_OUTPUT_SIZE>  Sl_mlp_2::generate_output() {
    idsp::c_vector<double, MLP_OUTPUT_SIZE> test_result = mlp.generate_output();
    // return mlp.generate_output();
    static idsp::c_vector<double, 3> previous_data_set = {0.0, 0.0, 0.0};
    idsp::c_vector<double, 3> found_data = mlp.find_closest_data_set(test_result);
     if (found_data[0] != previous_data_set [0] || found_data[1] != previous_data_set [1] || found_data[2] != previous_data_set [2]) {
        std::cout << "Found data set: " << found_data[0] << " " << found_data[1] << " " << found_data[2] << "\n";
        previous_data_set = found_data;
    }
    // std::cout << "Found data set: " << found_data[0] << " " << found_data[1] << " " << found_data[2] << "\n";
    return test_result;
}