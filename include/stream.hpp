#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <mutex>
#include <queue>
#include <atomic>
#include <memory>
#include "miniaudio.h"
#include "dr_wav.h"
#include "calculator.hpp"

class AudioStreamer {
public:
    AudioStreamer(int n_fft=512, int hop_len=256, int sr=16000, SE::Calculator* calculator=nullptr, bool saved=false);
    AudioStreamer(const std::string& src, int n_fft=512, int hop_len=256, int sr=16000, SE::Calculator* calculator=nullptr, bool saved=false);
    ~AudioStreamer();

    // 开始/结束音频流处理
    bool start();
    void stop();
    bool empty() { return !m_isRunning; }

private:
    // 实时流式音频数据回调（静态函数，通过pUserData访问实例）
    static void dataCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);

    std::unique_ptr<ma_device>   m_device;       // 音频设备
    std::unique_ptr<ma_decoder>  m_decoder;      // 音频设备
    std::unique_ptr<drwav>       m_saver_input;  // 录制器1
    std::unique_ptr<drwav>       m_saver_output; // 录制器2
    SE::Calculator*          m_calculator;   // 音频信号处理计算类
    std::atomic<bool>            m_isRunning;
    
    int m_nfft;
    int m_nhop;
    int m_sr;
    float* m_buffer;            // [0:255, 256:511]->[last, cur]
    float* m_processData;       // [256:511]->[cur]
    ma_uint64 m_samples;        // 音频文件总采样数
    float* m_tmp;               // 临时缓冲区
};