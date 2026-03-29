#pragma once
#ifdef ENABLE_RKNN

#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include "signal.h"
#include "rknn_api.h"
#include "backend.hpp"

class RKNNer_zerocpy : public Backend{
public:
    RKNNer_zerocpy(const char* rknn_file, const std::string fmt="nchw");
    ~RKNNer_zerocpy();

    SE::Vectorf forward(const SE::Vectorf& input) override;

private:

    void init_caches() override;
    void rearange();
    
    rknn_context m_ctx;
    rknn_sdk_version m_version;
    rknn_mem_size m_mem_size;
    std::string m_fmt;
    std::vector<rknn_tensor_mem*> m_caches;
    std::vector<rknn_tensor_mem*> m_caches_nhwc;
    rknn_tensor_mem* m_output;

    size_t m_input_count;
    std::vector<rknn_tensor_attr> m_input_attrs;
    std::vector<rknn_tensor_attr> m_native_input_attrs;
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
