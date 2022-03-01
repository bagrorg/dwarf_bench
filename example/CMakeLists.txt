if(ENABLE_DPCPP)
    add_example(lock_add lock_add.cpp)
    add_example(tbb_join tbb_join.cpp)

  target_link_libraries(tbb_join PRIVATE tbb)
endif()