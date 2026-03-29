#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include "miniaudio.h"

namespace SE{

inline std::vector<std::vector<float>> load(const std::string& path, ma_uint32 sr, float duration=0.f, ma_format dtype=ma_format_f32, bool verbose=true)
{
    // 1. 初始化解码器
    ma_decoder_config config = ma_decoder_config_init(
        dtype,      // 输出格式：32 位浮点
        1,          // 输出声道数（0 = 保持原文件的声道数）
        sr          // 输出采样率（0 = 保持原文件的采样率）
    );
    ma_decoder decoder;
    ma_result result = ma_decoder_init_file(path.c_str(), &config, &decoder);
    if (result != MA_SUCCESS) 
    {
        std::cerr << "Failed to load audio file" << std::endl;
        return std::vector<std::vector<float>>();
    }

    // 2. 获取音频信息
    ma_uint64 numChannels = decoder.outputChannels; // 声道数
    ma_uint64 totalFrames;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &totalFrames); // 总帧数
    ma_uint64 useFrames = duration<1e-8 ? totalFrames : duration*sr;
    
    // 3. 准备存储采样点的 vector (格式: vector[声道][采样点])
    std::vector<std::vector<float>> data(numChannels, std::vector<float>(useFrames, 0.f));

    // 4. 创建临时缓冲区（每次读取 1024 帧）
    const ma_uint32 bufferFrames = 1024;
    std::vector<float> buffer(bufferFrames * numChannels, 0.f);
    // float buffer[bufferFrames * numChannels] = {0}; // 缓冲区大小 = 帧数 × 声道数

    ma_uint64 framesRead;
    bool read_enough = false;
    ma_uint64 cur = 0;
    do {
        // 5. 读取音频数据到缓冲区
        result = ma_decoder_read_pcm_frames(&decoder, buffer.data(), bufferFrames, &framesRead);
        if (result != MA_SUCCESS && result != MA_AT_END) {
            std::cerr << "Error reading audio frames" << std::endl;
            break;
        }

        // 6. 将缓冲区数据拷贝到 vector
        for (ma_uint64 i = 0; i < framesRead; ++i) 
        {
            for (ma_uint32 ch = 0; ch < numChannels; ++ch) 
            {
                // 计算缓冲区中的索引（交错格式：frame0_ch0, frame0_ch1, frame1_ch0...）
                const ma_uint32 index = i * numChannels + ch;
                data[ch][cur++] = buffer[index];
            }
            if(cur==useFrames)
            {
                read_enough = true;
                break;
            }
        }
    } while (!read_enough  && framesRead > 0); // 循环直到读取完毕

    // 7. 关闭解码器
    ma_decoder_uninit(&decoder);

    // 输出统计信息
    if(verbose)
    {
        std::cout << "read \"" << path << "\" success!" << std::endl;
        std::cout << "|*********************************************************************|" << std::endl;
        std::cout << "sr          : " << sr << std::endl;
        std::cout << "channels    : " << numChannels << std::endl;
        std::cout << "samples     : " << totalFrames << std::endl;
        std::cout << "read samples: " << data[0].size() << std::endl;
        std::cout << "|*********************************************************************|" << std::endl;
    }

    return data;
}


}