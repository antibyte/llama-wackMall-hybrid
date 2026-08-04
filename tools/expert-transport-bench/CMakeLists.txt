find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

set(TARGET llama-expert-transport-bench)

add_executable(${TARGET} expert-transport-bench.cu)
target_link_libraries(${TARGET} PRIVATE CUDA::cudart)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
set_target_properties(${TARGET} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
