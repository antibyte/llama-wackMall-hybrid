set(TARGET llama-expert-compute-bench)

add_executable(${TARGET} expert-compute-bench.cpp)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${TARGET} PRIVATE llama)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
