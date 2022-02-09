add_tbb_lib(tbbsort tbbsort.cpp)
add_tbb_lib(permutation_buffer_sort permutation_buffer_sort.cpp)

if(ENABLE_DPCPP)
    add_dpcpp_lib(radix radix.cpp)
    if(ENABLE_CUDA)
        add_dpcpp_cuda_lib(radix radix_cuda.cpp)
    endif()
endif()