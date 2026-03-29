#include "onnxer.hpp"

const wchar_t* char_to_wchar(const char* str) 
{
    if (!str || *str == '\0') return L"";

    // 创建转换器
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    
    try {
        std::wstring wstr = converter.from_bytes(str);
        
        // 复制数据到堆内存
        wchar_t* wbuf = new wchar_t[wstr.size() + 1];
        wcscpy(wbuf, wstr.c_str());
        return wbuf;
    } catch (const std::range_error& e) {
        throw std::runtime_error(e.what());
    }
}

ONNXer::ONNXer(const char* onnx_file)
{
    // 1.初始化onnx推理器
    Ort::SessionOptions session_options;
    // session_options.AddConfigEntry("session.use_device_allocator_for_initializers", "1");   // 内存预分配
    // session_options.EnableCpuMemArena();      // 启用 CPU 内存池
    // session_options.EnableMemPattern();       // 启用内存模式优化
    session_options.SetIntraOpNumThreads(1);    // 单个算子并行
    session_options.SetInterOpNumThreads(1);    // 算子间流水线并行
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    m_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "");
#ifdef _WIN32
    const wchar_t* onnx_file_ort = char_to_wchar(onnx_file);
    m_session = std::make_unique<Ort::Session>(m_env, onnx_file_ort, session_options);
#else
    m_session = std::make_unique<Ort::Session>(m_env, onnx_file, session_options);
#endif
    // 2.获取模型信息
    Ort::AllocatorWithDefaultOptions allocator;
    m_input_count = m_session->GetInputCount();
    for(int i=0; i<m_input_count; i++)
    {
        // 获取input名称
        auto name_ptr = m_session->GetInputNameAllocated(i, allocator);
        char* name = new char[100];
        memset(name, 0, 100);
        strcpy(name, name_ptr.get());
        m_input_names.emplace_back(name);
        // 获取input形状
        Ort::TypeInfo type_info = m_session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        m_input_dims.emplace_back(tensor_info.GetShape());
        // 获取input大小
        m_input_sizes.emplace_back(tensor_info.GetElementCount());
    }
    m_output_count = m_session->GetOutputCount();
    for(int i=0; i<m_output_count; i++)
    {
        // 获取output名称
        auto name_ptr = m_session->GetOutputNameAllocated(i, allocator);
        char* name = new char[100];
        memset(name, 0, 100);
        strcpy(name, name_ptr.get());
        m_output_names.emplace_back(name);
        // 获取output形状
        Ort::TypeInfo type_info = m_session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        m_output_dims.emplace_back(tensor_info.GetShape());
        // 获取input大小
        m_output_sizes.emplace_back(tensor_info.GetElementCount());
    }
    
    init_caches();
}

ONNXer::~ONNXer()
{
    m_session->release();
    m_session.reset();
}

void ONNXer::init_caches()
{
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for(int i=0; i<m_input_count; i++)
    {
        m_raw_caches.emplace_back(std::vector<float>(m_input_sizes[i], 0));
        Ort::Value data_tensor = Ort::Value::CreateTensor<float>(memory_info, m_raw_caches[i].data(), m_input_sizes[i], m_input_dims[i].data(), m_input_dims[i].size());
        m_caches.push_back(std::move(data_tensor));
    }
}

SE::Vectorf ONNXer::forward(const SE::Vectorf& input)
{
    float* data_ptr = m_caches[0].GetTensorMutableData<float>();
    std::copy(input.begin(), input.end(), data_ptr);

    std::vector<Ort::Value> res = m_session->Run(Ort::RunOptions{nullptr}, m_input_names.data(), m_caches.data(), m_caches.size(), m_output_names.data(), m_output_count);
    for(int i=1; i<m_input_count; i++)
    {
        float* cache_ptr = m_caches[i].GetTensorMutableData<float>();
        float* output_ptr = res[i].GetTensorMutableData<float>();
        memcpy(cache_ptr, output_ptr, sizeof(float)*m_input_sizes[i]);
    }
    float* out_ptr = res[0].GetTensorMutableData<float>();

    return Eigen::Map<SE::Vectorf>(out_ptr, m_output_sizes[0]);
}