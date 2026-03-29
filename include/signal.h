#pragma once
#include "Eigen/Core"
#include "Eigen/Dense"
#include "unsupported/Eigen/FFT"

#include <vector>
#include <complex>
#include <iostream>
#include <unordered_map>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif // !M_PI

namespace SE{

static Eigen::FFT<float> fft;

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
#ifdef ENABLE_RKNN
typedef Eigen::Matrix<__fp16, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorh;
#endif
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixcf;

// 分/合帧
inline Eigen::MatrixXf frame(const Eigen::VectorXf& signal, int frame_length, int hop_length, const std::string& pad_mode = "constant")
{
    if (signal.size() == 0 || frame_length <= 0 || hop_length <= 0) 
        throw std::invalid_argument("Invalid input parameters.");

    const int n_samples = static_cast<int>(signal.size());
    const int n_frames = 1 + (n_samples - frame_length) / hop_length;
    const int required_samples = (n_frames - 1) * hop_length + frame_length;
    const int pad_size = (required_samples > n_samples) ? required_samples - n_samples : 0;

    Eigen::VectorXf padded_signal(n_samples + pad_size);
    padded_signal.head(n_samples) = signal;

    if (pad_size > 0) 
    {
        if (pad_mode == "reflect") 
        {
            for (int i = 0; i < pad_size; ++i) 
            {
                int idx = n_samples - 1 - (i % (n_samples - 1));
                padded_signal(n_samples + i) = signal(idx);
            }
        } 
        else 
            padded_signal.tail(pad_size).setZero();
    }

    return Eigen::Map<const Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>>(
        padded_signal.data(),
        frame_length,
        n_frames,
        Eigen::OuterStride<>(hop_length)
    );
}
inline Eigen::VectorXf deframe(const Eigen::MatrixXf& frames, int frame_length, int hop_length)
{
    // --- 初始化合成信号 ---
    const int num_frames = static_cast<int>(frames.cols());
    const int signal_length = (num_frames - 1) * hop_length + frame_length;
    Eigen::VectorXf synthesized_signal = Eigen::VectorXf::Zero(signal_length);

    // --- 计算重叠因子（归一化用）---
    float overlap_factor = 1.0f;
    if (hop_length < frame_length)
        overlap_factor = std::ceil(static_cast<float>(frame_length) / hop_length);

    // --- 预计算所有帧的起始位置 ---
    Eigen::VectorXi starts = Eigen::VectorXi::LinSpaced(num_frames, 0, (num_frames-1)*hop_length);
    Eigen::VectorXi ends = (starts.array() + frame_length).cwiseMin(signal_length);
    Eigen::VectorXi lengths = ends - starts;

    // --- 向量化叠加（利用Eigen的块操作和广播）---
    for (int i = 0; i < num_frames; ++i) 
    {
        Eigen::VectorXf frame = frames.col(i).head(lengths[i]);    // 提取当前帧（自动处理最后一帧截断）
        synthesized_signal.segment(starts[i], lengths[i]) += frame;    // 叠加到合成信号
    }

    // --- 归一化处理 ---
    if (hop_length < frame_length)
        synthesized_signal /= overlap_factor;

    return synthesized_signal;
}

// 窗函数
inline Vectorf hamming(int M, bool sym=false)
{
    bool needs_trunc = false;
    if(!sym)
    {
        M += 1;
        needs_trunc = true;
    }

    Eigen::VectorXf n = Eigen::VectorXf::LinSpaced(M, float(-3.1415926535), float(3.1415926535));
    Eigen::VectorXf w(M);
    w = 0.54 + 0.46*Eigen::cos(n.array());

    return needs_trunc? w.segment(0, M-1) : w;
}
inline Vectorf triang(int M, bool sym=false) 
{
    bool needs_trunc = false;
    if(!sym)
    {
        M += 1;
        needs_trunc = true;
    }
    const bool is_even = (M % 2 == 0);
    const int half = (M + 1) / 2;  // 计算半长

    // 生成 n = [1, 2, ..., half]
    Eigen::VectorXf n = Eigen::VectorXf::LinSpaced(half, 1.f, static_cast<float>(half));
    Eigen::VectorXf w;

    if (is_even) 
    {
        // 偶数情况: w = (2n - 1) / M
        w = (2.0f * n.array() - 1.0f) / static_cast<float>(M);
        // 拼接 w 和逆序的 w
        const int new_size = 2 * half;
        w.conservativeResize(new_size);
        w.tail(half) = w.head(half).reverse();
    } 
    else 
    {
        // 奇数情况: w = 2n / (M + 1)
        w = 2.0f * n.array() / static_cast<float>(M + 1);
        // 拼接 w 和 w[-2::-1] (去掉最后一个元素的逆序)
        const int new_size = 2 * half - 1;
        w.conservativeResize(new_size);
        if (half > 1)
            w.tail(half - 1) = w.head(half - 1).reverse();
    }

    return needs_trunc? w.segment(0, M-1) : w;
}
inline Vectorf hann(int M, bool sym=false)
{
    return 0.5*(1.f-(Vectorf::LinSpaced(M, 0.f, static_cast<float>(M-1))*2.f*M_PI/M).array().cos());
}
using windowFuncType = Vectorf (*)(int , bool);
static const std::unordered_map<std::string, windowFuncType> windows_map = {
    {"triang", triang},
    {"hamming", hamming},
    {"hann", hann}
};
inline Eigen::MatrixXf add_window(const Eigen::MatrixXf& frames, std::string window="hamming")
{
    std::unordered_map<std::string, windowFuncType>::const_iterator window_pair;
    if((window_pair=windows_map.find(window)) == windows_map.end())
        window = "hamming";
    Eigen::VectorXf window_seq = window_pair->second(static_cast<int>(frames.rows()), false);

    return frames.array().colwise() * window_seq.array();
}
inline Eigen::VectorXf remove_window(const Eigen::VectorXf& signal, int win_length, std::string window="hamming")
{
    std::unordered_map<std::string, windowFuncType>::const_iterator window_pair;
    if((window_pair=windows_map.find(window)) == windows_map.end())
       return signal;
    Eigen::VectorXf window_seq = window_pair->second(static_cast<int>(win_length), false);
    float gain = 1.f / window_seq.mean();
    return gain * signal;
}

inline Vectorf pad(const Vectorf &x, int left, int right, const std::string &mode, float value=0.f)
{
    Vectorf x_paded = Vectorf::Constant(left+x.size()+right, value);
    x_paded.segment(left, x.size()) = x;
  
    if (mode.compare("reflect") == 0)
    {
        for (int i = 0; i < left; ++i)
            x_paded[i] = x[left-i];
        for (int i = left; i < left+right; ++i)
            x_paded[i+x.size()] = x[x.size()-2-i+left];
    }
    if (mode.compare("symmetric") == 0)
    {
        for (int i = 0; i < left; ++i)
            x_paded[i] = x[left-i-1];
        for (int i = left; i < left+right; ++i)
            x_paded[i+x.size()] = x[x.size()-1-i+left];
    }
    if (mode.compare("edge") == 0)
    {
        for (int i = 0; i < left; ++i)
            x_paded[i] = x[0];
        for (int i = left; i < left+right; ++i)
            x_paded[i+x.size()] = x[x.size()-1];
    }
    return x_paded;
}

inline Matrixcf stft(const Vectorf &x, int n_fft, int n_hop, const std::string& win, bool center, const std::string &mode)
{
    // window
    std::unordered_map<std::string, windowFuncType>::const_iterator window_pair;
    if((window_pair=windows_map.find(win)) == windows_map.end())
    {
        printf("Error:no %s function.", win.c_str());
        exit(-1);
    }
    Vectorf window = window_pair->second(static_cast<int>(x.size()), false);
  
    int pad_len = center ? n_fft / 2 : 0;
    Vectorf x_paded = pad(x, pad_len, pad_len, mode, 0.f);
  
    int n_f = n_fft/2+1;
    int n_frames = 1+(x_paded.size()-n_fft) / n_hop;
    Matrixcf X(n_frames, n_fft);
    // Eigen::FFT<float> fft;
  
    for (int i = 0; i < n_frames; ++i)
    {
        Vectorf x_frame = window.array()*x_paded.segment(i*n_hop, n_fft).array();
        X.row(i) = fft.fwd(x_frame);
    }
    return X.leftCols(n_f);
}
inline Matrixcf stft(const Vectorf &x, int n_fft, int n_hop, const Vectorf& win, bool center, const std::string &mode)
{
    // window
    const Vectorf& window = win;
  
    int pad_len = center ? n_fft / 2 : 0;
    Vectorf x_paded = pad(x, pad_len, pad_len, mode, 0.f);
  
    int n_f = n_fft/2+1;
    int n_frames = 1+(x_paded.size()-n_fft) / n_hop;
    Matrixcf X(n_frames, n_fft);
    // Eigen::FFT<float> fft;
  
    for (int i = 0; i < n_frames; ++i)
    {
        Vectorf x_frame = window.array()*x_paded.segment(i*n_hop, n_fft).array();
        X.row(i) = fft.fwd(x_frame);
    }
    return X.leftCols(n_f);
}
inline std::vector<std::vector<std::complex<float>>> stft_stl(std::vector<float> &x, int n_fft, int n_hop, const std::string& win, bool center, const std::string &mode)
{
    Vectorf map_x = Eigen::Map<Vectorf>(x.data(), x.size());
    Matrixcf X = stft(map_x, n_fft, n_hop, win, center, mode);
    std::vector<std::vector<std::complex<float>>> X_vector(X.rows(), std::vector<std::complex<float>>(X.cols(), 0));
    for (int i = 0; i < X.rows(); ++i)
    {
        auto &row = X_vector[i];
        Eigen::Map<Vectorcf>(row.data(), row.size()) = X.row(i);
    }
    return X_vector;
}
inline std::vector<std::vector<std::complex<float>>> stft_stl(std::vector<float> &x, int n_fft, int n_hop, const Vectorf& win, bool center, const std::string &mode)
{
    Vectorf map_x = Eigen::Map<Vectorf>(x.data(), x.size());
    Matrixcf X = stft(map_x, n_fft, n_hop, win, center, mode);
    std::vector<std::vector<std::complex<float>>> X_vector(X.rows(), std::vector<std::complex<float>>(X.cols(), 0));
    for (int i = 0; i < X.rows(); ++i)
    {
        auto &row = X_vector[i];
        Eigen::Map<Vectorcf>(row.data(), row.size()) = X.row(i);
    }
    return X_vector;
}

inline SE::Vectorf irfft(const SE::Matrixcf& spec, int nfft)
{
    // 构建spec的对称共轭部分：X[N−k]=X*[k], k=1,2,...,⌊N/2⌋−1
    SE::Matrixcf spec_conjugate = spec.block(0, 1, 1, spec.cols()-2).conjugate().reverse();    
    SE::Matrixcf spec_full(spec.rows(), spec.cols()+spec_conjugate.cols());
    spec_full << spec, spec_conjugate;

    Eigen::VectorXf tmp;
    SE::fft.inv(tmp, Eigen::VectorXcf(spec_full.transpose()), nfft);   //补全共轭部分

    return tmp.transpose();
}


inline Matrixf power2db(const Matrixf& x) 
{
    auto log_sp = 10.0f*x.array().max(1e-10).log10();
    return log_sp.cwiseMax(log_sp.maxCoeff() - 80.0f);
}

inline Matrixf spectrogram(const Matrixcf &X, float power = 1.f)
{
    return X.cwiseAbs().array().pow(power);
}

inline Matrixf dct(const Matrixf& x, bool norm, int type) 
{
    int N = x.cols();
    Matrixf xi = Matrixf::Zero(N, N);
    xi.rowwise() += Vectorf::LinSpaced(N, 0.f, static_cast<float>(N-1));
    // type 2
    Matrixf coeff = 2*(M_PI*xi.transpose().array()/N*(xi.array()+0.5)).cos();
    Matrixf dct = x*coeff.transpose();
    // ortho
    if (norm) 
    {
        Vectorf ortho = Vectorf::Constant(N, std::sqrt(0.5f/N));
        ortho[0] = std::sqrt(0.25f/N);
        dct = dct*ortho.asDiagonal();
    }
    return dct;
}

inline Matrixf melfilter(int sr, int n_fft, int n_mels, int fmin, int fmax, bool norm=true, bool htk=false)
{
    int n_f = n_fft/2+1;
    Vectorf fft_freqs = (Vectorf::LinSpaced(n_f, 0.f, static_cast<float>(n_f-1))*sr)/n_fft;
  
    float f_min = 0.f;
    float f_sp = 200.f/3.f;
    float min_log_hz = 1000.f;
    float min_log_mel = (min_log_hz-f_min)/f_sp;
    float logstep = logf(6.4f)/27.f;
  
    auto hz_to_mel = [=](int hz, bool htk = false) -> float 
                    {
                        if (htk)
                            return 2595.0f*log10f(1.0f+hz/700.0f);
                        float mel = (hz-f_min)/f_sp;
                        if (hz >= min_log_hz)
                            mel = min_log_mel+logf(hz/min_log_hz)/logstep;
                        return mel;
                    };
    auto mel_to_hz = [=](Vectorf &mels, bool htk = false) -> Vectorf 
                    {
                        if (htk)
                            return 700.0f*(Vectorf::Constant(n_mels+2, 10.f).array().pow(mels.array()/2595.0f)-1.0f);
                        return (mels.array()>min_log_mel).select(((mels.array()-min_log_mel)*logstep).exp()*min_log_hz, (mels*f_sp).array()+f_min);
                    };
  
    float min_mel = hz_to_mel(fmin, htk);
    float max_mel = hz_to_mel(fmax, htk);
    Vectorf mels = Vectorf::LinSpaced(n_mels+2, min_mel, max_mel);
    Vectorf mel_f = mel_to_hz(mels, htk);
    Vectorf fdiff = mel_f.segment(1, mel_f.size() - 1) - mel_f.segment(0, mel_f.size() - 1);
    Matrixf ramps = mel_f.replicate(n_f, 1).transpose().array() - fft_freqs.replicate(n_mels + 2, 1).array();
  
    Matrixf lower = -ramps.topRows(n_mels).array()/fdiff.segment(0, n_mels).transpose().replicate(1, n_f).array();
    Matrixf upper = ramps.bottomRows(n_mels).array()/fdiff.segment(1, n_mels).transpose().replicate(1, n_f).array();
    Matrixf weights = (lower.array()<upper.array()).select(lower, upper).cwiseMax(0);
  
    if(norm)
    {
        auto enorm = (2.0/(mel_f.segment(2, n_mels)-mel_f.segment(0, n_mels)).array()).transpose().replicate(1, n_f);
        weights = weights.array()*enorm;
    }
  
    return weights;
}

inline Matrixf melspectrogram_kernel(const Vectorf &x, int sr, int n_fft, int n_hop,
    const std::string &win, bool center,
    const std::string &mode, float power,
    int n_mels, int fmin, int fmax, bool norm=true, bool htk=false)
{
    // Matrixcf X = stft(x, n_fft, n_hop, win, center, mode);
    Matrixcf X = stft(x, n_fft, n_hop, win, center, mode);
    static Matrixf mel_basis = melfilter(sr, n_fft, n_mels, fmin, fmax, norm, htk); // static only to accelerate(have bug!)
    Matrixf sp = spectrogram(X, power);
    Matrixf mel = mel_basis*sp.transpose();
    return mel;
}

inline Matrixf melspectrogram(const Vectorf &x, int sr, 
    int n_fft, int n_hop, const std::string &win, bool center, const std::string &mode,
    float power, int n_mels, int fmin, int fmax, bool norm=true, bool htk=false)
{
    Matrixf mel = melspectrogram_kernel(x, sr, n_fft, n_hop, win, center, mode, power, n_mels, fmin, fmax, norm, htk).transpose();
    return mel;
}

inline std::vector<std::vector<float>> melspectrogram_stl(std::vector<float> &x, int sr, 
    int n_fft, int n_hop, const std::string &win, bool center, const std::string &mode,
    float power, int n_mels, int fmin, int fmax, bool norm=true, bool htk=false)
{
    Vectorf map_x = Eigen::Map<Vectorf>(x.data(), x.size());
    Matrixf mel = melspectrogram_kernel(map_x, sr, n_fft, n_hop, win, center, mode, power, n_mels, fmin, fmax, norm, htk).transpose();
    std::vector<std::vector<float>> mel_vector(mel.rows(), std::vector<float>(mel.cols(), 0.f));
    for (int i = 0; i < mel.rows(); ++i)
    {
        auto &row = mel_vector[i];
        Eigen::Map<Vectorf>(row.data(), row.size()) = mel.row(i);
    }
    return mel_vector;
}

inline Matrixf mfcc(const Vectorf &x, int sr,
    int n_fft, int n_hop, const std::string &win, bool center, const std::string &mode,
    float power, int n_mels, int fmin, int fmax,
    int n_mfcc, bool norm, int type, bool mel_norm=true, bool htk=false) 
{
    Matrixf mel = melspectrogram_kernel(x, sr, n_fft, n_hop, win, center, mode, power, n_mels, fmin, fmax, mel_norm, htk).transpose();
    Matrixf mel_db = power2db(mel);
    Matrixf dct = SE::dct(mel_db, norm, type).leftCols(n_mfcc);
    return dct;
}

inline std::vector<std::vector<float>> mfcc_stl(std::vector<float> &x, int sr,
    int n_fft, int n_hop, const std::string &win, bool center, const std::string &mode,
    float power, int n_mels, int fmin, int fmax,
    int n_mfcc, bool norm, int type, bool mel_norm=true, bool htk=false) 
{
    Vectorf map_x = Eigen::Map<Vectorf>(x.data(), x.size());
    Matrixf mel = melspectrogram_kernel(map_x, sr, n_fft, n_hop, win, center, mode, power, n_mels, fmin, fmax, mel_norm, htk).transpose();
    Matrixf mel_db = power2db(mel);
    Matrixf dct = SE::dct(mel_db, norm, type).leftCols(n_mfcc);
    std::vector<std::vector<float>> mfcc_vector(dct.rows(), std::vector<float>(dct.cols(), 0.f));
    for (int i = 0; i < dct.rows(); ++i) 
    {
        auto &row = mfcc_vector[i];
        Eigen::Map<Vectorf>(row.data(), row.size()) = dct.row(i);
    }
    return mfcc_vector;
}

// 子空间滤波器实现
inline Eigen::MatrixXf toeplitz(const Eigen::VectorXf& c) 
{
    const int n = static_cast<int>(c.size());
    Eigen::MatrixXf mat(n, n);
    
    // 转换为原生数组指针（可选优化，减少 Eigen 封装开销）
    const float* c_data = c.data();
    
    // 按列优先存储优化内存访问
    float val = 0.f;
    for (int col = 0; col < n; ++col) 
    {
        for (int row = 0; row <= col; ++row) 
        {  // 仅计算上三角
            val = c_data[col - row];
            mat(row, col) = val;  // 列优先写入
            mat(col, row) = val;  // 对称赋值
        }
    }
    
    return mat;
}
inline Eigen::VectorXf xcorr(const Eigen::VectorXf& x, int p)
{
    int n = static_cast<int>(x.size());
    if(p > n)
    {
        p = n;
        std::cout << "warning: p > x.size() -> let p=x.size()." << std::endl;
    }
    Eigen::VectorXf x_pad(n+n-1);
    x_pad = Eigen::VectorXf::Zero(n+n-1);
    x_pad.segment(0, n) = x;
    Eigen::VectorXf r(p); 
    r = Eigen::VectorXf::Zero(p);
    for(int i=0; i<p; i++)
        r[i] = x.dot(x_pad.segment(i, n));
    return r;
}
inline Eigen::VectorXf filterFunc(const Eigen::VectorXf& x, float lowbound=0.f, float upbound=20.f, float min_w=0.7f, float max_w=2.f)
{
    float k = (max_w-min_w) / (upbound-lowbound+float(1e-12));
    float b = max_w - k*upbound;

    // 分段函数向量化实现
    Eigen::VectorXf y=Eigen::VectorXf::Zero(x.size()); 
    y = (x.array()<lowbound).select(max_w, y);  // 下界
    y = (x.array()>upbound).select(min_w, y);   // 上界
    y = ((x.array()<=upbound)&&(x.array()>=lowbound)).select(k*x.array()+b, y); // kx+b

    return y;
}
inline Eigen::VectorXf SSFilter(const Eigen::VectorXf& noisy_frame, const Eigen::MatrixXf& Rn)
{
    int p = static_cast<int>(Rn.cols());

    // 1.进行协方差矩阵 Ry 估计
    Eigen::MatrixXf Ry = toeplitz(xcorr(noisy_frame, p));

    // 2.计算iRnRx
    Eigen::MatrixXf In = Eigen::MatrixXf::Identity(p, p);
    Eigen::MatrixXf RnI = (In*1e-8) + Rn;
    Eigen::MatrixXf iRn = RnI.inverse();
    Eigen::MatrixXf iRnRx = iRn*Ry - In;

    // 3.获取 特征向量 V与对角线特征值矩阵 D
    Eigen::EigenSolver<Eigen::MatrixXf> solver(iRnRx);
    Eigen::VectorXf d = solver.eigenvalues().real();
    Eigen::MatrixXf V = solver.eigenvectors().real();
    Eigen::MatrixXf iV_T = V.completeOrthogonalDecomposition().pseudoInverse().transpose();

    // 4.计算增益和映射矩阵H
    d = d.cwiseMax(0.0f);
    Eigen::VectorXf SNR_db = 10.f * Eigen::log10(d.array()+1e-12);
    Eigen::VectorXf mu = filterFunc(SNR_db, 0.f, 20.f, 0.7f, 2.0f);
    Eigen::VectorXf gain_vals= d.array() / (d+mu).array();
    Eigen::MatrixXf G= gain_vals.asDiagonal();
    Eigen::MatrixXf H = iV_T * G * V.transpose();

    // 5.语音增强
    int hop_len = int(p/2);
    std::string window = "triang";
    Eigen::MatrixXf sub_frames = frame(noisy_frame, p, hop_len);
    Eigen::MatrixXf enh_sub_frames = H * sub_frames;
    enh_sub_frames = add_window(enh_sub_frames, window);
    Eigen::VectorXf enh_signal = deframe(enh_sub_frames, p, hop_len);
    enh_signal = remove_window(enh_signal, static_cast<int>(enh_signal.size()), window);

    // 6.补上尾部
    int noisy_length = static_cast<int>(noisy_frame.size());
    int enh_length = static_cast<int>(enh_signal.size());
    if(enh_length < noisy_length)
    {
        Eigen::VectorXf tail_frame;
        tail_frame = Eigen::VectorXf::Zero(p);
        tail_frame.segment(0, noisy_length-enh_length) = noisy_frame.segment(enh_length, noisy_length-enh_length);
        Eigen::VectorXf tail_signal = (H * tail_frame).segment(0, noisy_length-enh_length);
        enh_signal.conservativeResize(enh_signal.size()+tail_signal.size());
        enh_signal.segment(enh_signal.size(), tail_signal.size()) = tail_signal;
    }
 
    return enh_signal;
}


}

