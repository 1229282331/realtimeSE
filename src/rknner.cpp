#include "rknner.hpp"

#ifdef ENABLE_RKNN

static void nchw_to_nhwc_kernel(const __fp16* input, __fp16* output,
    std::size_t N, std::size_t C,
    std::size_t H, std::size_t W) 
{
    const std::size_t HW = H * W;
    // Parallelize over batches and rows for speed
    // #pragma omp parallel for collapse(2)
    for (std::size_t n = 0; n < N; ++n) 
    {
        for (std::size_t h = 0; h < H; ++h) 
        {
            for (std::size_t w = 0; w < W; ++w) 
            {
                // Compute output base index: ((n * H + h) * W + w) * C
                std::size_t out_base = ((n * H + h) * W + w) * C;
                // Compute input base index: (n * C) * HW + h * W + w
                std::size_t in_base  = n * (C * HW) + h * W + w;
                // Loop over channels (innermost for contiguous writes)
                for (std::size_t c = 0; c < C; ++c) 
                    output[out_base + c] = input[in_base + c * HW];
            }
        }
    }
}

void RKNNer::nchw_to_nhwc()
{
    for(int i=0; i<m_caches.size(); i++)
    {
        if(m_input_dims[i].size()==4)
        {
            int n=m_input_dims[i][0]; 
            int h=m_input_dims[i][1]; 
            int w=m_input_dims[i][2]; 
            int c=m_input_dims[i][3]; 
            nchw_to_nhwc_kernel(reinterpret_cast<const __fp16*>(m_caches[i].buf), reinterpret_cast<__fp16*>(m_caches_nhwc[i].buf), n, c, h, w);
        }
        else if(m_input_dims[i].size()==3)
        {
            int n=1; 
            int c=m_input_dims[i][0]; 
            int w=m_input_dims[i][1]; 
            int h=m_input_dims[i][2]; 
            nchw_to_nhwc_kernel(reinterpret_cast<const __fp16*>(m_caches[i].buf), reinterpret_cast<__fp16*>(m_caches_nhwc[i].buf), n, c, h, w);
        }
    }
}

RKNNer::RKNNer(const char* rknn_file, const std::string fmt)
{
    // 1.初始化rknn推理器
    m_fmt = fmt;
    int ret = rknn_init(&m_ctx, const_cast<char*>(rknn_file), 0, 0, NULL);
    ret = rknn_query(m_ctx, RKNN_QUERY_SDK_VERSION, &m_version, sizeof(m_version));
    ret = rknn_query(m_ctx, RKNN_QUERY_MEM_SIZE, &m_mem_size, sizeof(m_mem_size));
    rknn_input_output_num io_num;
    ret = rknn_query(m_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    m_input_count = io_num.n_input;
    m_output_count = io_num.n_output;
    m_input_attrs = std::vector<rknn_tensor_attr>(m_input_count);
    m_output_attrs = std::vector<rknn_tensor_attr>(m_output_count);
    for (uint32_t i = 0; i < m_input_count; i++) 
    {
        m_input_attrs[i].index = i;
        ret = rknn_query(m_ctx, RKNN_QUERY_INPUT_ATTR, &(m_input_attrs[i]), sizeof(rknn_tensor_attr));
        m_input_names.emplace_back(m_input_attrs[i].name);
        m_input_dims.emplace_back(std::vector<long>(m_input_attrs[i].dims, m_input_attrs[i].dims+m_input_attrs[i].n_dims));
        m_input_sizes.emplace_back(m_input_attrs[i].n_elems);
    }
    for (uint32_t i = 0; i < m_output_count; i++) 
    {
        m_output_attrs[i].index = i;
        ret = rknn_query(m_ctx, RKNN_QUERY_OUTPUT_ATTR, &(m_output_attrs[i]), sizeof(rknn_tensor_attr));
        m_output_names.emplace_back(m_output_attrs[i].name);
        m_output_dims.emplace_back(std::vector<long>(m_output_attrs[i].dims, m_output_attrs[i].dims+m_output_attrs[i].n_dims));
        m_output_sizes.emplace_back(m_output_attrs[i].n_elems);
    }
    printf("rknn_api/rknnrt version: %s, driver version: %s\n", m_version.api_version, m_version.drv_version);
    printf("total weight size: %dKB, total internal size: %dKB\n", m_mem_size.total_weight_size/1024, m_mem_size.total_internal_size/1024);
    printf("total dma used size: %zuKB\n", (size_t)m_mem_size.total_dma_allocated_size/1024);

    init_caches();
}

RKNNer::~RKNNer()
{
    rknn_destroy(m_ctx);
}

void RKNNer::init_caches()
{
    if(m_fmt == "nchw")
        for(int i=0; i<m_input_count; i++)
        {
            m_raw_caches.emplace_back(std::vector<__fp16>(m_input_sizes[i], 0));
            m_raw_caches_nhwc.emplace_back(std::vector<__fp16>(m_input_sizes[i], 0));
            rknn_input input, input_nhwc;
            input.index        = i;
            input.pass_through = 0;
            input.type         = m_input_attrs[i].type;
            input.fmt          = m_input_attrs[i].fmt;
            input.buf          = m_raw_caches[i].data();
            input.size         = m_input_sizes[i] * sizeof(__fp16);
            input_nhwc.index        = i;
            input_nhwc.pass_through = 0;
            input_nhwc.type         = m_input_attrs[i].type;
            input_nhwc.fmt          = m_input_attrs[i].fmt;
            input_nhwc.buf          = m_raw_caches_nhwc[i].data();
            input_nhwc.size         = m_input_sizes[i] * sizeof(__fp16);
            m_caches.push_back(input);
            m_caches_nhwc.push_back(input_nhwc);
        }
    else
        for(int i=0; i<m_input_count; i++)
        {
            m_raw_caches.emplace_back(std::vector<__fp16>(m_input_sizes[i], 0));
            rknn_input input;
            input.index        = i;
            input.pass_through = 0;
            input.type         = m_input_attrs[i].type;
            input.fmt          = m_input_attrs[i].fmt;
            input.buf          = m_raw_caches[i].data();
            input.size         = m_input_sizes[i] * sizeof(__fp16);
            m_caches.push_back(input);
            m_caches_nhwc.push_back(input);
        }

    m_outputs = std::vector<rknn_output>(m_output_count);
    m_outputs[0].want_float  = 1;
    m_outputs[0].index       = 0;
    m_outputs[0].is_prealloc = 0;
    for (uint32_t i = 1; i < m_output_count; ++i) 
    {
        m_outputs[i].want_float  = 0;
        m_outputs[i].index       = i;
        m_outputs[i].is_prealloc = 1;
        m_outputs[i].buf = m_caches[i].buf;
        m_outputs[i].size = m_output_sizes[i] * sizeof(__fp16);
    }
}


SE::Vectorf RKNNer::forward(const SE::Vectorf& input)
{
    
    SE::Vectorh input_half = input.cast<__fp16>();
    __fp16* data_ptr = reinterpret_cast<__fp16*>(m_caches[0].buf);
    std::copy(input_half.begin(), input_half.end(), data_ptr);

    if(m_fmt == "nchw")
    {
        nchw_to_nhwc();
        rknn_inputs_set(m_ctx, m_input_count, m_caches_nhwc.data());
        rknn_run(m_ctx, NULL);
        rknn_outputs_get(m_ctx, m_output_count, m_outputs.data(), NULL);
    }
    else
    {
        rknn_inputs_set(m_ctx, m_input_count, m_caches.data());
        rknn_run(m_ctx, NULL);
        rknn_outputs_get(m_ctx, m_output_count, m_outputs.data(), NULL);   
    }
    float* out_ptr = reinterpret_cast<float*>(m_outputs[0].buf);
    
    return Eigen::Map<SE::Vectorf>(out_ptr, m_output_sizes[0]);
}

#endif