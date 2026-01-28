cmake_minimum_required(VERSION 3.21)
project(rocm_attn LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HIP REQUIRED)

set(HIP_DEVICE_TARGET "")
if (TARGET hip::device)
    set(HIP_DEVICE_TARGET hip::device)
elseif (TARGET HIP::device)
    set(HIP_DEVICE_TARGET HIP::device)
elseif (TARGET hip::amdhip64)
    set(HIP_DEVICE_TARGET hip::amdhip64)
elseif (TARGET HIP::amdhip64)
    set(HIP_DEVICE_TARGET HIP::amdhip64)
endif()

set_source_files_properties(src/attn_kernel.hip.cpp PROPERTIES LANGUAGE HIP)

add_library(rocm_attn STATIC
    src/attn_kernel.hip.cpp
)

target_include_directories(rocm_attn PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${HIP_INCLUDE_DIRS}
)

if (HIP_DEVICE_TARGET)
    target_link_libraries(rocm_attn PUBLIC ${HIP_DEVICE_TARGET})
elseif (HIP_LIBRARIES)
    target_link_libraries(rocm_attn PUBLIC ${HIP_LIBRARIES})
endif()

option(BUILD_TESTS "Build tests" ON)
if (BUILD_TESTS)
    enable_testing()
    set_source_files_properties(src/tests/attn_test.cpp PROPERTIES LANGUAGE HIP)
    add_executable(attn_test
        src/tests/attn_test.cpp
    )
    target_link_libraries(attn_test PRIVATE rocm_attn)
    if (HIP_DEVICE_TARGET)
        target_link_libraries(attn_test PRIVATE ${HIP_DEVICE_TARGET})
    elseif (HIP_LIBRARIES)
        target_link_libraries(attn_test PRIVATE ${HIP_LIBRARIES})
    endif()
    add_test(NAME attn_test COMMAND attn_test)
endif()
