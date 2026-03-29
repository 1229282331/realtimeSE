#pragma once

#include <complex>
#include <chrono>
#include "signal.h"
#include "backend.hpp"
#include "onnxer.hpp"
#include "rknner.hpp"
#include "rknner_zerocpy.hpp"
#include "onnxruntime_cxx_api.h"

namespace SE{

SE::Vectorf subspace_pred(const SE::Vectorf& input, Backend& session, int nfft=512, int hop_len=256, int p=64, int target_p=64, int sr=16000);
Matrixcf unet_pred(const Matrixcf& input_spec, const Matrixcf& preenhanced_spec, Backend& session);

class Calculator{
public:
    Calculator(int nfft=512, int hop_len=256, const std::string& model1_file="", const std::string& model2_file="");
    ~Calculator();
    
    SE::Vectorf forward(SE::Vectorf& input);

private:
    int m_nfft;
    int m_hop_len;
    int m_pad_len;
    SE::Vectorf m_win;
    SE::Vectorf m_norm;
    bool m_first_frame;
    SE::Vectorf m_last;

    std::unique_ptr<Backend> m_session1;
    std::unique_ptr<Backend> m_session2;

    std::string m_backend1;
    std::string m_backend2;
};
    


}