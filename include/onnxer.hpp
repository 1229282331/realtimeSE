#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <locale>
#include <codecvt>
#include "onnxruntime_cxx_api.h"
#include "signal.h"
#include "backend.hpp"

class ONNXer : public Backend{
public:
    ONNXer(const char* onnx_file);
    ~ONNXer();

    SE::Vectorf forward(const SE::Vectorf& input) override;

private:
    void init_caches() override;
    
    Ort::Env m_env;
    std::unique_ptr<Ort::Session> m_session;
    std::vector<std::vector<float>> m_raw_caches;   //生命周期与类对象相同
    std::vector<Ort::Value> m_caches;

    size_t m_input_count;
    std::vector<const char*> m_input_names;
    std::vector<std::vector<int64_t>> m_input_dims;
    std::vector<int64_t> m_input_sizes;
    size_t m_output_count;
    std::vector<const char*> m_output_names;
    std::vector<std::vector<int64_t>> m_output_dims;
    std::vector<int64_t> m_output_sizes;
};
