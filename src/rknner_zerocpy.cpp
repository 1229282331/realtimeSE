#include "rknner_zerocpy.hpp"

#ifdef ENABLE_RKNN

static void NCHW_to_NHWC_kernel(const __fp16* input, __fp16* output,
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

void NCHW_to_NC1HWC2_kernel(const __fp16* src, __fp16* dst, int N, int C, int H, int W, int C2) 
{
    const int C1 = (C+C2-1) / C2;
    const int PADDED_C = C1 * C2;

    for (int n = 0; n < N; ++n) 
        for (int h = 0; h < H; ++h) 
            for (int w = 0; w < W; ++w) 
                for (int c1 = 0; c1 < C1; ++c1) 
                    for (int c2 = 0; c2 < C2; ++c2) 
                    {
                        // 计算原始通道索引
                        const int src_c = c1 * C2 + c2;
                        
                        // 计算目标索引
                        const int dst_idx = n * (C1 * H * W * C2) + c1 * (H * W * C2) + h * (W * C2) + w * C2 + c2;
                        
                        // 计算源索引
                        const int src_idx = n * (C * H * W) + src_c * (H * W) + h * W + w;

                        // 带边界检查的赋值
                        if (src_c < C) 
                            dst[dst_idx] = src[src_idx];
                        else 
                            dst[dst_idx] = 0; // 填充0
                    }
}


void RKNNer_zerocpy::rearange()
{
    for(int i=0; i<m_caches.size(); i++)
    {
        if(m_native_input_attrs[i].fmt==RKNN_TENSOR_NC1HWC2)
        {
            int n=m_input_dims[i][0]; 
            int h=m_input_dims[i][1]; 
            int w=m_input_dims[i][2]; 
            int c=m_input_dims[i][3]; 
            NCHW_to_NC1HWC2_kernel(reinterpret_cast<const __fp16*>(m_caches[i]->virt_addr), reinterpret_cast<__fp16*>(m_caches_nhwc[i]->virt_addr), n, c, h, w, m_native_input_attrs[i].dims[4]);
        }
        else if(m_native_input_attrs[i].fmt==RKNN_TENSOR_NHWC)
        {
            int n=m_input_dims[i][0]; 
            int h=m_input_dims[i][1]; 
            int w=m_input_dims[i][2]; 
            int c=m_input_dims[i][3]; 
            NCHW_to_NHWC_kernel(reinterpret_cast<const __fp16*>(m_caches[i]->virt_addr), reinterpret_cast<__fp16*>(m_caches_nhwc[i]->virt_addr), n, c, h, w);
        }
        else if(m_native_input_attrs[i].fmt==RKNN_TENSOR_UNDEFINED)
        {
            int n=1; 
            int c=m_input_dims[i][0]; 
            int w=m_input_dims[i][1]; 
            int h=m_input_dims[i][2]; 
            NCHW_to_NHWC_kernel(reinterpret_cast<const __fp16*>(m_caches[i]->virt_addr), reinterpret_cast<__fp16*>(m_caches_nhwc[i]->virt_addr), n, c, h, w);
        }
    }
}

RKNNer_zerocpy::RKNNer_zerocpy(const char* rknn_file, const std::string fmt)
{
    // 1.初始化rknn推理器
    m_fmt = fmt;
    int ret = rknn_init(&m_ctx, const_cast<char*>(rknn_file), 0, RKNN_FLAG_MODEL_BUFFER_ZERO_COPY, NULL);
    ret = rknn_query(m_ctx, RKNN_QUERY_SDK_VERSION, &m_version, sizeof(m_version));
    ret = rknn_query(m_ctx, RKNN_QUERY_MEM_SIZE, &m_mem_size, sizeof(m_mem_size));
    rknn_input_output_num io_num;
    ret = rknn_query(m_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    m_input_count = io_num.n_input;
    m_output_count = io_num.n_output;
    m_input_attrs = std::vector<rknn_tensor_attr>(m_input_count);
    m_output_attrs = std::vector<rknn_tensor_attr>(m_output_count);
    m_native_input_attrs = std::vector<rknn_tensor_attr>(m_input_count);
    for (uint32_t i = 0; i < m_input_count; i++) 
    {
        m_input_attrs[i].index = i;
        m_native_input_attrs[i].index = i;
        ret = rknn_query(m_ctx, RKNN_QUERY_INPUT_ATTR, &(m_input_attrs[i]), sizeof(rknn_tensor_attr));
        ret = rknn_query(m_ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(m_native_input_attrs[i]), sizeof(rknn_tensor_attr));
        m_input_attrs[i].pass_through = 1;
        m_input_attrs[i].fmt = m_native_input_attrs[i].fmt;
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

RKNNer_zerocpy::~RKNNer_zerocpy()
{
    for(int i=0; i <m_caches.size(); i++)
    {
        if(m_caches[i]!=nullptr)
            rknn_destroy_mem(m_ctx,m_caches[i]);
        if(m_caches_nhwc[i]!=nullptr)
            rknn_destroy_mem(m_ctx,m_caches_nhwc[i]);
    }
    if(m_output != nullptr)
        rknn_destroy_mem(m_ctx,m_output);

    rknn_destroy(m_ctx);
}

void RKNNer_zerocpy::init_caches()
{
    if(m_fmt == "nchw")
        for(int i=0; i<m_input_count; i++)
        {
            rknn_tensor_mem *tmp1=nullptr, *tmp2 = nullptr;//表示输入和输出内存信息的两个变量
            tmp1 = rknn_create_mem(m_ctx, m_input_attrs[i].size_with_stride * sizeof(__fp16));       // 申请输入内存，表示补齐了无效像素后，实际存储图像数据所占用内存空间的大小
            tmp2 = rknn_create_mem(m_ctx, m_native_input_attrs[i].size_with_stride * sizeof(__fp16));       // 申请输入内存，表示补齐了无效像素后，实际存储图像数据所占用内存空间的大小
            memset(tmp1->virt_addr, 0, m_input_attrs[i].size_with_stride * sizeof(__fp16));
            memset(tmp2->virt_addr, 0, m_native_input_attrs[i].size_with_stride * sizeof(__fp16));
            m_caches.emplace_back(tmp1);
            m_caches_nhwc.emplace_back(tmp2);
            rknn_set_io_mem(m_ctx, m_caches_nhwc[i], &m_input_attrs[i]);
        }
    else
        for(int i=0; i<m_input_count; i++)
        {
            rknn_tensor_mem *tmp=nullptr;
            tmp = rknn_create_mem(m_ctx, m_input_sizes[i] * sizeof(__fp16));       // 申请输入内存，表示补齐了无效像素后，实际存储图像数据所占用内存空间的大小
            memset(tmp->virt_addr, 0, m_input_sizes[i] * sizeof(__fp16));
            m_caches.emplace_back(tmp);
            m_caches_nhwc.emplace_back(tmp);

            rknn_set_io_mem(m_ctx, m_caches_nhwc[i], &m_input_attrs[i]);
        }

    m_output = rknn_create_mem(m_ctx, m_output_sizes[0] * sizeof(float));
    m_output_attrs[0].type = RKNN_TENSOR_FLOAT32;
    rknn_set_io_mem(m_ctx, m_output, &m_output_attrs[0]);
    for (uint32_t i = 1; i < m_output_count; ++i) 
    {
        m_output_attrs[i].type = RKNN_TENSOR_FLOAT16;
        rknn_set_io_mem(m_ctx, m_caches[i], &m_output_attrs[i]);
    }
}

SE::Vectorf RKNNer_zerocpy::forward(const SE::Vectorf& input)
{
    SE::Vectorh input_half = input.cast<__fp16>();
    __fp16* data_ptr = reinterpret_cast<__fp16*>(m_caches[0]->virt_addr);
    std::copy(input_half.begin(), input_half.end(), data_ptr);

    if(m_fmt == "nchw")
        rearange();
    rknn_run(m_ctx, NULL);
    float* out_ptr = reinterpret_cast<float*>(m_output->virt_addr);
    
    return Eigen::Map<SE::Vectorf>(out_ptr, m_output_sizes[0]);
}

#endif
