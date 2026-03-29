#pragma once
#ifdef ENABLE_RKNN

#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include "signal.h"
#include "rknn_api.h"
#include "backend.hpp"

class RKNNer : public Backend{
public:
    RKNNer(const char* rknn_file, const std::string fmt="nchw");
    ~RKNNer();

    SE::Vectorf forward(const SE::Vectorf& input) override;

private:

    void init_caches() override;
    void nchw_to_nhwc();
    
    rknn_context m_ctx;
    rknn_sdk_version m_version;
    rknn_mem_size m_mem_size;
    std::string m_fmt;
    std::vector<std::vector<__fp16>> m_raw_caches;   //生命周期与类对象相同
    std::vector<std::vector<__fp16>> m_raw_caches_nhwc;   //生命周期与类对象相同
    std::vector<rknn_input> m_caches;
    std::vector<rknn_input> m_caches_nhwc;
    std::vector<rknn_output> m_outputs;

    size_t m_input_count;
    std::vector<rknn_tensor_attr> m_input_attrs;
    std::vector<const char*> m_input_names;
    std::vector<std::vector<long>> m_input_dims;
    std::vector<long> m_input_sizes;
    size_t m_output_count;
    std::vector<rknn_tensor_attr> m_output_attrs;
    std::vector<const char*> m_output_names;
    std::vector<std::vector<long>> m_output_dims;
    std::vector<long> m_output_sizes;
};

#endif