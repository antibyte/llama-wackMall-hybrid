set(TARGET llama-turboquant-ref)

add_executable(${TARGET}
    main.cpp
    turboquant-ref.cpp
)
target_include_directories(${TARGET} PRIVATE ../../ggml/include)
target_link_libraries(${TARGET} PRIVATE ggml-base)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
