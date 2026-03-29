# realtimeSE源码



### 目录结构

```shell
├── audio												# 测试音频(clean/noisy)
│   └── clnsp1_train_69005_1_snr15_tl-21_fileid_158/		
│   |	├── clean.wav			    					# 纯净语音音频  
│   |	├── noisy.wav									# 带噪语音音频
│   └──      ...			  
│   └── ght												# 广哈实际场景带噪音频		 
├── model												# 训练保存的模型参数文件
│   └── onnx/	  										# onnx模型文件(CPU)
│   |	├── nenet.onnx			    					# 第一阶段子空间噪声估计网络  
│   |	├── spcrn_att_conv.onnx			    			# 第二阶段基于自注意力-卷积的语音增强网络 
│   |	└── spcrn_rnn_wconv.onnx			  			# 第二阶段基于LSTM-小波卷积的语音增强网络
│   └── rknn/	  										# rknn模型文件(NPU)   
│   |	├── nenet_rk3568.rknn			    			# 计算延时<1ms  
│   |	├── nenet_rk3588.rknn			    			# 计算延时<1ms     
│   |	├── spcrn_att_conv_rk3568.rknn			    	# attention并行度高，精度保持不变，计算延时9-10ms(3568)   
│   |	├── spcrn_att_conv_rk3588.rknn			    	# attention并行度高，精度保持不变，计算延时4-5ms(3588单核) 
│   |	├── spcrn_rnn_wconv_rk3568.rknn			    	# 模型对fp32->fp16精度敏感，频域LSTM无法并行，[不可用]!  
│   |	└── spcrn_rnn_wconv_rk3588.rknn			  		# 模型对fp32->fp16精度敏感，频域LSTM无法并行，[不可用]!
├── scripts/											# 数据处理脚本
│   ├── eval.py			 								# 评估降噪前后SISNR，PESQ指标
│   ├── evaluation_func.py								# 基础语音增强/降噪评估函数
│   └── plot.py			 								# 离线绘制波形/语谱图
├── 3rdparty/											# 第三方库源码/预编译 
│   └── dr_libs/										# 轻量级音频加载库（源码）
│   └── eigen/											# 高性能线性数学库（源码）
│   └── miniaudio/			  							# 轻量级音频IO处理库（源码）
│   └── onnxruntime/									# onnxruntime推理库（预编译）
│   └── rknpu2/											# rknn推理库（预编译）
│   ├── CMakeLists.txt			    					# 3rdparty作为子目录即时编译
├── include/											# 项目源码头文件
│   |── backend.hpp										# 推理后端硬件设备抽象类（当前支持CPU、RKNPU）
│   |── calculator.hpp									# 推理计算类，封装了完整的推理计算过程
│   |── load_ndarray.hpp								# 加载/保存.npy
│   |── loader.hpp										# 加载音频函数实现
│   |── onnxer.hpp										# 推理后端派生类ONNX，实现完整CPU推理
│   |── rknner_zerocpy.hpp								# 推理后端派生类RKNN，实现完整RKNN推理零拷贝+自定义tensor布局转换
│   |── rknner.hpp										# 推理后端派生类RKNN，实现完整RKNN推理，rknn api自动转换（非常慢）
│   |── saver.hpp										# 保存音频函数实现
│   |── signal.hpp										# 所有音频数学处理函数实现（等效于python实现）
│   |── stream.hpp										# 实时流式推理类，负责流式推理逻辑
├── src/												# 项目源码
│   └── cnpy/											# 第三方读写.npy实现
│   └── wavreader/										# 第三方读取.wav实现
│   |── calculator.cpp									# calculator.hpp对应源码实现
│   |── onnxer.cpp										# onnxer.hpp对应源码实现
│   |── rknner_zerocpy.cpp								# rknner_zerocpy.hpp对应源码实现
│   |── rknner.cpp										# rknner.hpp对应源码实现
│   |── stream.cpp										# stream.hpp对应源码实现
│── CMakeLists.txt										# ENABLE_RKNN=ON时开启RKNPU支持，在无NPU设备上需设置OFF
|── main.cpp											# 可执行程序源码
```

### 使用

#### 1. 环境准备

- 安装python环境（**可选**）

```shell
# 仅用于运行scripts/下的数据处理分析脚本，自行pip install xxx所需库
```

#### 2. CPU,DDR,NPU定频（针对rk3568/3588）

- CPU开启**性能**模式

```shell
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor		# "policy0"可能不一样
```

- DDR开启**性能**模式

```shell
echo performance > /sys/class/devfreq/dmc/governor
```

- NPU开启**性能**模式

```shell
echo performance > /sys/class/devfreq/fdab0000.npu/governor						# "fdab0000.npu"可能不一样
```

#### 3.构建&编译

```shell
> mkdir build
> cd build
> cmake ..
> cmake --build .
```

- 若要编译rknpu加速版本，需设置cmake构建选项**ENABLE_RKNN=ON**，否则CPU版本应设置**ENABLE_RKNN=OFF**
- 考虑到buildroot系统可能没有Fortran编译器，本项目关闭了eigen对于Fortran加速blas的配置。最好编译buildroot系统时添加对fortran的支持，打开eigen fortran blas以提高CPU计算效率。
- 上述构建&编译命令**默认编译debug版本**，release优化版本需额外设置。推荐使用vscode cmake插件进行构建编译，方便指定编译器和编译版本。

#### 4. 运行

```shell
realtimeSE [-m] <第一阶段模型> <第二阶段模型> [-i] <输入音频文件.wav>
```

- [-m] 指定选用的降噪模型，可选参数。不设置[-m]时为**透传**，不进行任何降噪。
- [-i]   指定输入音频文件noisy.wav，可选参数。不设置时输入设备为麦克风。

* **注**：运行后可随时中止程序，但必须输入**q+回车**才能合法退出程序，否则（ctrl+c）会导致实时录制的.wav文件末尾错误。

```shell
taskset -c 0 realtimeSE [-m] <第一阶段模型> <第二阶段模型> [-i] <输入音频文件.wav>		# 指定使用CPU第0号小核运行
```

#### 4. 示例

- 麦克风->扬声器，透传模式：

```shell
./realtimeSE
```

- 麦克风->扬声器，仅开启第一阶段子空间降噪模式：

```shell
./realtimeSE -m model/onnx/nenet.onnx			# 运行在CPU上
./realtimeSE -m model/rknn/nenet_rk3568.rknn	# 运行在RK3568 NPU上
```

- 麦克风->扬声器，开启完整降噪模式（第一阶段+第二阶段）：

```shell
./realtimeSE -m model/onnx/nenet.onnx model/onnx/spcrn_att_conv.onnx				# model1,model2均运行在CPU上
./realtimeSE -m model/onnx/nenet.onnx model/rknn/spcrn_att_conv_rk3568.rknn			# model1运行在CPU上，model2运行在NPU上
./realtimeSE -m model/rknn/nenet_rk3568.rknn model/rknn/spcrn_att_conv_rk3568.rknn	# model1,model2均运行在RK3568NPU上
```

- 文件->扬声器，开启完整降噪模式（第一阶段+第二阶段）：

```shell
./realtimeSE -m model/rknn/nenet_rk3568.rknn model/rknn/spcrn_att_conv_rk3568.rknn -i audio/ght/noisy2.wav
```

#### 5.结果

​	程序默认开启实时录制功能，将实时记录的**输入音频**，降噪处理后的**输出音频**分别保存为**input.wav**，**output.wav**

#### 6. 数据分析

- 评估降噪效果

```shell
python scripts/eval.py <纯净语音.wav> <带噪语音.wav> <降噪语音.wav>
```

- 绘制音频波形/语谱图

```shell
python scripts/plot.py <1.wav> <2.wav> ... <n.wav>
```

- 查看CPU各核心占用率

```shell
mpstat -P ALL 1  # 每隔1秒刷新一次
```

- 查看NPU占用率

```shell
watch -n 0.1 sudo cat /sys/kernel/debug/rknpu/load
```

