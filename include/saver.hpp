#pragma once
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include <string>
#include <vector>
#include <chrono>

namespace SE{

inline void save(const std::string& file, const std::vector<std::vector<float>>& data, int sr)
{
    // 1. 配置音频参数
    const int sampleRate = sr;
    const int numChannels = data.size();
    const int numFrames = data[0].size();  // 假设所有声道长度相同

    // 2. 将 vector<vector<float>> 转换为交错格式的 vector<float>
    std::vector<float> interleaved(numFrames * numChannels);
    for (size_t i = 0; i < numFrames; ++i) 
        for (int ch = 0; ch < numChannels; ++ch) 
            interleaved[i * numChannels + ch] = data[ch][i];

    // 3. 初始化 dr_wav写入器
    drwav wav;
    drwav_data_format format;
    format.container = drwav_container_riff;     // WAV 文件格式
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;   // 32 位浮点
    format.channels = numChannels;
    format.sampleRate = sampleRate;
    format.bitsPerSample = 32;                   // 必须与 format.format 匹配

    if (!drwav_init_file_write(&wav, file.c_str(), &format, NULL)) 
    {
        std::cerr << "Failed to open writting file" << std::endl;
        return;
    }

    // 4. 写入数据到文件
    drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, numFrames, interleaved.data());

    // 5. 检查写入是否完整
    if (framesWritten != numFrames) {
        // 写入失败（磁盘空间不足？）
        drwav_uninit(&wav);
        return;
    }

    // 6. 关闭写入器
    drwav_uninit(&wav);
}

}