#include "stream.hpp"

AudioStreamer::AudioStreamer(int n_fft, int hop_len, int sr, SE::Calculator* calculator, bool saved) 
    : m_isRunning(false), m_nfft(n_fft), m_nhop(hop_len), m_sr(sr), m_samples(0), m_calculator(calculator)
{
    // 初始化录制器
    if(saved)
    {
        m_saver_input = std::make_unique<drwav>();
        m_saver_output = std::make_unique<drwav>();
        drwav_data_format format;
        format.container = drwav_container_riff;     // WAV 文件格式
        format.format = DR_WAVE_FORMAT_IEEE_FLOAT;   // 32 位浮点
        format.channels = 1;
        format.sampleRate = m_sr;
        format.bitsPerSample = 32;                   // 必须与 format.format 匹配
        if (!drwav_init_file_write(m_saver_input.get(), "input.wav", &format, NULL)) 
        {
            std::cerr << "Failed to open writting file" << std::endl;
            m_saver_input.reset();
        }
        if (!drwav_init_file_write(m_saver_output.get(), "output.wav", &format, NULL)) 
        {
            std::cerr << "Failed to open writting file" << std::endl;
            m_saver_output.reset();
        }
    }

    // 初始化音频设备配置
    ma_device_config deviceConfig   = ma_device_config_init(ma_device_type_duplex); // 双工模式（输入+输出）
    deviceConfig.capture.format     = ma_format_f32;     // 输入格式：32位浮点
    deviceConfig.playback.format    = ma_format_f32;     // 输出格式：32位浮点
    deviceConfig.sampleRate         = m_sr;              // 采样率
    deviceConfig.capture.channels   = 1;                 // 输入声道数（单声道）
    deviceConfig.playback.channels  = 1;                 // 输出声道数（立体声）
    deviceConfig.periodSizeInFrames = m_nhop;            // 设置每帧采集的采样点
    deviceConfig.dataCallback       = dataCallback;      // 数据回调
    deviceConfig.pUserData          = this;              // 传递this指针到回调

    // 初始化音频设备
    m_device = std::make_unique<ma_device>();
    if (ma_device_init(nullptr, &deviceConfig, m_device.get()) != MA_SUCCESS) 
    {
        std::cerr << "Failed to initialize audio device" << std::endl;
        return;
    }

    // 初始化缓冲区
    m_tmp = nullptr;
    m_buffer = new float[m_nfft];
    m_processData = new float[m_nhop];
    memset(m_buffer, 0, m_nfft*sizeof(float));
    memset(m_processData, 0, m_nhop*sizeof(float));
}

AudioStreamer::AudioStreamer(const std::string& src, int n_fft, int hop_len, int sr, SE::Calculator* calculator, bool saved)
    : m_isRunning(false), m_nfft(n_fft), m_nhop(hop_len), m_sr(sr), m_samples(0), m_calculator(calculator)
{
    // 初始化录制器
    if(saved)
    {
        m_saver_input = std::make_unique<drwav>();
        m_saver_output = std::make_unique<drwav>();
        drwav_data_format format;
        format.container = drwav_container_riff;     // WAV 文件格式
        format.format = DR_WAVE_FORMAT_IEEE_FLOAT;   // 32 位浮点
        format.channels = 1;
        format.sampleRate = m_sr;
        format.bitsPerSample = 32;                   // 必须与 format.format 匹配
        if (!drwav_init_file_write(m_saver_input.get(), "input.wav", &format, NULL)) 
        {
            std::cerr << "Failed to open writting file" << std::endl;
            m_saver_input.reset();
        }
        if (!drwav_init_file_write(m_saver_output.get(), "output.wav", &format, NULL)) 
        {
            std::cerr << "Failed to open writting file" << std::endl;
            m_saver_output.reset();
        }
    }

    // 初始化解码器
    m_decoder = std::make_unique<ma_decoder>();
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, sr);
    ma_result result = ma_decoder_init_file(src.c_str(), &config, m_decoder.get());
    if (result != MA_SUCCESS) 
    {
        std::cerr << "Failed to load audio file" << std::endl;
        exit(-1);
    }
    ma_uint64 numChannels = m_decoder->outputChannels; // 声道数
    result = ma_decoder_get_length_in_pcm_frames(m_decoder.get(), &m_samples); // 总帧数

    // 初始化缓冲区
    m_tmp = new float[m_nhop];
    m_buffer = new float[m_nfft];
    m_processData = new float[m_nhop];
    memset(m_tmp, 0, m_nhop*sizeof(float));
    memset(m_buffer, 0, m_nfft*sizeof(float));
    memset(m_processData, 0, m_nhop*sizeof(float));

    // 初始化音频设备配置
    ma_device_config deviceConfig   = ma_device_config_init(ma_device_type_playback); // 双工模式（输入+输出）
    deviceConfig.playback.format    = ma_format_f32;     // 输出格式：32位浮点
    deviceConfig.sampleRate         = m_sr;              // 采样率
    deviceConfig.playback.channels  = 1;                 // 输出声道数（立体声）
    deviceConfig.periodSizeInFrames = m_nhop;            // 设置每帧采集的采样点
    deviceConfig.dataCallback       = dataCallback;      // 数据回调
    deviceConfig.pUserData          = this;              // 传递this指针到回调

    // 初始化音频设备
    m_device = std::make_unique<ma_device>();
    if (ma_device_init(nullptr, &deviceConfig, m_device.get()) != MA_SUCCESS) 
    {
        std::cerr << "Failed to initialize audio device" << std::endl;
        return;
    }
}

AudioStreamer::~AudioStreamer() 
{
    stop();
    ma_device_uninit(m_device.get());
    ma_decoder_uninit(m_decoder.get());
    drwav_uninit(m_saver_input.get());
    drwav_uninit(m_saver_output.get());
    m_device.reset();
    m_decoder.reset();
    m_saver_input.reset();
    m_saver_output.reset();
    m_calculator = nullptr;
    delete []m_buffer;
    delete []m_processData;
    if(m_tmp)
        delete []m_tmp;
}

bool AudioStreamer::start() 
{
    if (ma_device_start(m_device.get()) != MA_SUCCESS) 
    {
        std::cerr << "Failed to start audio device" << std::endl;
        return false;
    }
    memset(m_buffer, 0, m_nfft*sizeof(float));
    memset(m_processData, 0, m_nhop*sizeof(float));
    if(m_tmp)
        memset(m_tmp, 0, m_nhop*sizeof(float));
    m_isRunning = true;
    return true;
}

void AudioStreamer::stop() 
{
    if (m_isRunning) 
    {
        ma_device_stop(m_device.get());
        m_isRunning = false;
    }
}

// 音频数据回调（静态函数，通过pUserData访问实例）
void AudioStreamer::dataCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) 
{
    // 注意：只考虑单通道！
    AudioStreamer* self = static_cast<AudioStreamer*>(pDevice->pUserData);
    float* output = static_cast<float*>(pOutput);
    const float* input = static_cast<const float*>(pInput);
    if(self->m_decoder.get())
    {
        ma_result result = ma_decoder_read_pcm_frames(self->m_decoder.get(), self->m_tmp, self->m_nhop, (ma_uint64*)(&frameCount));
        if (result != MA_SUCCESS && result != MA_AT_END) 
        {
            std::cerr << "Error reading audio frames" << std::endl;
            exit(-1);
        }
        input = self->m_tmp;
    }
    if(frameCount==0)
    {
        self->m_isRunning = false;
        return;
    }

    memcpy(self->m_buffer+self->m_nhop, input, frameCount * sizeof(float));     // 拷贝current frame

    auto start = std::chrono::steady_clock::now();
    // 处理音频数据（此处可添加自定义处理逻辑，如滤波、混音等）
    if(!self->m_calculator)
        memcpy(self->m_processData, self->m_buffer+self->m_nhop, sizeof(float)*frameCount);
    else
    {
        SE::Vectorf x = Eigen::Map<SE::Vectorf>(self->m_buffer, self->m_nfft);
        SE::Vectorf y = self->m_calculator->forward(x);
        memcpy(self->m_processData, y.data(), sizeof(float)*frameCount);
    }
    
    memcpy(self->m_buffer, input, frameCount * sizeof(float));  // 更新last frame
    memcpy(output, self->m_processData, sizeof(float)*frameCount);

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "cost: " << duration.count() << " ms" << std::endl;

    // 实时录制
    if(self->m_saver_input.get())
        drwav_uint64 save_samples = drwav_write_pcm_frames(self->m_saver_input.get(), frameCount, input);
    if(self->m_saver_output.get())
        drwav_uint64 save_samples = drwav_write_pcm_frames(self->m_saver_output.get(), frameCount, self->m_processData);
}

