set(TARGET llama-expert-compute-bench)

add_executable(${TARGET} expert-compute-bench.cpp)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(${TARGET} PRIVATE llama)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    target_sources(${TARGET} PRIVATE q4k-avx2.cu)
    target_link_libraries(${TARGET} PRIVATE CUDA::cudart)
    set_target_properties(${TARGET} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
