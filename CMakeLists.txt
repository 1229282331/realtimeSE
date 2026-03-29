cmake_minimum_required(VERSION 3.6)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/bin")
add_subdirectory(3rdparty)

project(realtimeSE)
set(TARGET_NAME ${CMAKE_PROJECT_NAME})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")


# 1. 创建配置选项
option(ENABLE_RKNN "Enable RKNN backend" OFF)  # 默认OFF

# ********************************** 设置宏 ********************************** 
if(ENABLE_RKNN)
    add_compile_definitions(ENABLE_RKNN)  # 或 target_compile_definitions
endif()

set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/3rdparty")

# GCC 专属优化
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # add_compile_options(
    #     -march=native
    #     -mtune=native
    # )

    # 启用快速数学（可选）
    # add_compile_options(-Ofast)

    # 添加编译/链接选项
    # add_compile_options(-pg)
    # add_link_options(-pg)

endif()

# 设置onnxruntime库
if(UNIX)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(ONNXRUNTIME_INCLUDE ${THIRD_PARTY_DIR}/onnxruntime/onnxruntime-linux-x64/include)
        set(ONNXRUNTIME_LIB ${THIRD_PARTY_DIR}/onnxruntime/onnxruntime-linux-x64/lib/libonnxruntime.so)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(ARM_ENABLE_NEON ON)
        set(ONNXRUNTIME_INCLUDE ${THIRD_PARTY_DIR}/onnxruntime/onnxruntime-linux-aarch64/include)
        set(ONNXRUNTIME_LIB ${THIRD_PARTY_DIR}/onnxruntime/onnxruntime-linux-aarch64/lib/libonnxruntime.so)
    else()
        set(ONNXRUNTIME_INCLUDE ${THIRD_PARTY_DIR}/onnxruntime/onnxruntime-osx-arm64/include)
        set(ONNXRUNTIME_LIB ${THIRD_PARTY_DIR}/onnxruntime/onnxruntime-osx-arm64/lib/libonnxruntime.1.21.0.dylib)
    endif()
elseif(WIN32)
    set(ONNXRUNTIME_INCLUDE ${THIRD_PARTY_DIR}/onnxruntime/onnxruntime-win-x64/include)
    set(ONNXRUNTIME_LIB ${THIRD_PARTY_DIR}/onnxruntime/onnxruntime-win-x64/lib/onnxruntime.lib)
    set(ONNXRUNTIME_SHARED ${THIRD_PARTY_DIR}/onnxruntime/onnxruntime-win-x64/lib/onnxruntime.dll)
endif()

# ********************************** 设置依赖库 **********************************  
if(ENABLE_RKNN)
    set(RKNNRT_LIB ${THIRD_PARTY_DIR}/rknpu2/lib/aarch64/librknnrt.so)
    set(RKNNRT_INCLUDE ${THIRD_PARTY_DIR}/rknpu2/include)
endif()
set(EIGEN_INCLUDE ${THIRD_PARTY_DIR}/eigen)
# set(FFTW_INCLUDE ${THIRD_PARTY_DIR}/fftw-3.3.10/api)
set(MINIAUDIO_INCLUDE ${THIRD_PARTY_DIR}/miniaudio)
set(DR_INCLUDE ${THIRD_PARTY_DIR}/dr_libs)


# ********************************************************************************

# 设置项目源代码和头文件
file(GLOB_RECURSE HEADER_FILES "*.h" "*.hpp")
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.c")

add_executable(${TARGET_NAME} main.cpp ${SOURCE_FILES} ${HEADER_FILES})


# ********************************** Link & Include **********************************  
if(ENABLE_RKNN)
    target_link_libraries(${TARGET_NAME} PUBLIC ${RKNNRT_LIB})
endif()
target_link_libraries(${TARGET_NAME} PUBLIC miniaudio)
# target_link_libraries(${TARGET_NAME} PUBLIC dr_libs_interface)    # 单头文件
target_link_libraries(${TARGET_NAME} PRIVATE ${ONNXRUNTIME_LIB})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${THIRD_PARTY_DIR}>)
if(ENABLE_RKNN)
    target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${RKNNRT_INCLUDE}>)
endif()
target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${EIGEN_INCLUDE}>)
target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${MINIAUDIO_INCLUDE}>)
target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${DR_INCLUDE}>)
target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${ONNXRUNTIME_INCLUDE}>)
# ********************************************************************************

if(WIN32)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                    ${ONNXRUNTIME_SHARED}
                    $<TARGET_FILE_DIR:${TARGET_NAME}>
                COMMENT "Copying DLL to output directory"
            )
endif()

# install target and libraries
# set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_benchmark_${CMAKE_SYSTEM_NAME})
# install(TARGETS rknn_benchmark DESTINATION ./)
# install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
