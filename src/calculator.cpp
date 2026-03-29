#include "calculator.hpp"

namespace SE{

SE::Vectorf subspace_pred(const SE::Vectorf& input, Backend& session, int nfft, int hop_len, int p, int target_p, int sr)
{
    target_p = std::min(target_p, p);

    // 1.特征提取
    SE::Vectorf noisy_data = SE::Vectorf::Zero(128);
    noisy_data.segment(0, 64) = SE::xcorr(input, p);
    noisy_data.segment(64, 64) = SE::mfcc(input, sr, nfft, hop_len, "hann", false, "reflect", 2.f, 80, 0, sr/2, 64, true, 2, false, true);
    float noisy_power = noisy_data.segment(0, 64).norm();

    // 2.模型推理
    SE::Vectorf rn = session.forward(noisy_data);
    float noise_power = Eigen::Map<SE::Vectorf>(rn.data(), rn.size()).norm();
    float subspace_snr = (noisy_power-noise_power) / (noise_power+1e-14);
    // 3.子空间滤波
    if(subspace_snr > 4.f)
        return input;
    else
        return SE::SSFilter(input, SE::toeplitz(rn.segment(0, target_p)));
}

Matrixcf unet_pred(const Matrixcf& input_spec, const Matrixcf& preenhanced_spec, Backend& session)
{
    Matrixcf enhanced_spec = Matrixcf::Zero(input_spec.rows(), input_spec.cols());
    
    // 1.特征提取
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> mixture_spec(4, input_spec.cols()); // 列主序以匹配onnx输入[1, 257, 1, 4]内存排布
    mixture_spec << input_spec.real(), input_spec.imag(), preenhanced_spec.real(), preenhanced_spec.imag();
    
    // 2.模型推理
    auto res = session.forward(Eigen::Map<SE::Vectorf>(mixture_spec.data(), mixture_spec.size()));
    Matrixf res_spec = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(res.data(), 2, 257);

    enhanced_spec = res_spec.row(0).cast<std::complex<float>>() + res_spec.row(1).cast<std::complex<float>>()*std::complex<float>(0.f, 1.f);
    return enhanced_spec;
}

Calculator::Calculator(int nfft, int hop_len, const std::string& model1_file, const std::string& model2_file) : m_nfft(nfft), m_hop_len(hop_len) , m_pad_len(nfft/2), m_first_frame(true)
{ 
    m_win = Eigen::pow(SE::hann(nfft).array(), 0.5f);
    SE::Vectorf win_sp = Eigen::pow(m_win.array(), 2.f);
    m_norm = win_sp.segment(0, hop_len).array() + win_sp.segment(hop_len, hop_len).array();
    m_last = SE::Vectorf::Zero(1, nfft);
    if(model1_file.size())
    {
        if(std::string(model1_file.begin()+model1_file.size()-4, model1_file.end()) == "onnx")
            m_session1 = std::make_unique<ONNXer>(model1_file.c_str());
#ifdef ENABLE_RKNN
        else if(std::string(model1_file.begin()+model1_file.size()-4, model1_file.end()) == "rknn")
            m_session1 = std::make_unique<RKNNer_zerocpy>(model1_file.c_str(), "nhwc");
#endif
        else
        {
            std::cout << "error: unknow model format!" << std::endl;
            exit(-1);
        }
    }
    if(model2_file.size())
    {
        if(std::string(model2_file.begin()+model2_file.size()-4, model2_file.end()) == "onnx")
            m_session2 = std::make_unique<ONNXer>(model2_file.c_str());
#ifdef ENABLE_RKNN
        else if(std::string(model2_file.begin()+model2_file.size()-4, model2_file.end()) == "rknn")
            m_session2 = std::make_unique<RKNNer_zerocpy>(model2_file.c_str(), "nchw");
#endif
        else
        {
            std::cout << "error: unknow model format!" << std::endl;
            exit(-1);
        }
    }
}

Calculator::~Calculator()
{
    m_session2.reset();
    m_session1.reset();
}

SE::Vectorf Calculator::forward(SE::Vectorf& input)
{
    // ************************** 第一阶段：子空间降噪 **************************
    if(m_session1.get() == nullptr) return input.segment(m_hop_len, m_hop_len);
    SE::Vectorf preenhanced = subspace_pred(input, *m_session1, m_nfft, m_hop_len, 64, 16);

    // input.array() *= 0.5f;
    // preenhanced.array() *= 0.5f;
    // ************************** 第二阶段：混合语音增强 **************************
    if(m_session2.get() == nullptr) return preenhanced.segment(m_hop_len, m_hop_len);
    
    SE::Matrixcf noisy_spec, preenhanced_spec;
    if(m_first_frame)
    {
        // noisy_spectrum
        SE::Vectorf tmp1 = input.segment(m_hop_len, m_hop_len);
        tmp1 = SE::pad(tmp1, m_pad_len-1, 0, "reflect", 0.f);
        SE::Vectorf input_pad = SE::Vectorf::Zero(1, m_nfft);
        input_pad.row(0).col(0) = tmp1.row(0).col(tmp1.cols()-2);
        input_pad.row(0).segment(1, tmp1.cols()).array() = tmp1.row(0).array();
        noisy_spec = SE::stft(input_pad, m_nfft, m_hop_len, m_win, false, "reflect");
        // preenhanced_spectrum
        SE::Vectorf tmp2 = preenhanced.segment(m_hop_len, m_hop_len);
        tmp2 = SE::pad(tmp2, m_pad_len-1, 0, "reflect", 0.f);
        SE::Vectorf preenhanced_pad = SE::Vectorf::Zero(1, m_nfft);
        preenhanced_pad.row(0).col(0) = tmp2.row(0).col(tmp2.cols()-2);
        preenhanced_pad.row(0).segment(1, tmp2.cols()).array() = tmp2.row(0).array();
        preenhanced_spec = SE::stft(preenhanced_pad, m_nfft, m_hop_len, m_win, false, "reflect");
    }
    else
    {
        noisy_spec = SE::stft(input, m_nfft, m_hop_len, m_win, false, "reflect");
        preenhanced_spec = SE::stft(preenhanced, m_nfft, m_hop_len, m_win, false, "reflect");
    }
    Matrixcf enhanced_spec = unet_pred(noisy_spec, preenhanced_spec, *m_session2);


    SE::Vectorf cur = SE::irfft(enhanced_spec, m_nfft).array() * m_win.array();

    cur.segment(0, m_hop_len).array() += m_last.segment(m_hop_len, m_hop_len).array();
    if(!m_first_frame)
        cur.segment(0, m_hop_len).array() /= m_norm.cwiseMax(1e-10).array();
    SE::Vectorf enhanced = cur.array();
    m_last = cur;
    m_first_frame = false;

    return enhanced.segment(0, m_hop_len);
}

}