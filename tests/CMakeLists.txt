enable_testing()

# TODO: add add_dpcpp_test() func

add_executable(scan_tests scan_tests.cpp)
add_executable(hash_table_tests hash_table_tests.cpp)
add_executable(join_tests join_tests.cpp)
add_executable(cuckoo_hashtable_tests cuckoo_hashtable_tests.cpp)
if(ENABLE_EXPERIMENTAL)
  add_executable(slab_tests slab_tests.cpp)
endif()

target_link_libraries(scan_tests gtest standalone_scan oclhelpers::oclhelpers)
target_link_libraries(hash_table_tests dpcpp_common sycl GTest::gtest)
target_link_libraries(cuckoo_hashtable_tests dpcpp_common sycl GTest::gtest)
target_link_libraries(join_tests join_helpers_lib sycl GTest::gtest)
if(ENABLE_EXPERIMENTAL)
  target_link_libraries(slab_tests dpcpp_common sycl GTest::gtest)
endif()



target_include_directories(scan_tests PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(hash_table_tests PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(cuckoo_hashtable_tests PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(join_tests PRIVATE ${PROJECT_SOURCE_DIR})
if(ENABLE_EXPERIMENTAL)
  target_include_directories(slab_tests PRIVATE ${PROJECT_SOURCE_DIR})
endif()

target_compile_options(scan_tests PRIVATE -fsycl)
target_link_options(scan_tests PRIVATE -fsycl)

target_compile_options(hash_table_tests PRIVATE -fsycl)
target_link_options(hash_table_tests PRIVATE -fsycl)

target_compile_options(cuckoo_hashtable_tests PRIVATE -fsycl)
target_link_options(cuckoo_hashtable_tests PRIVATE -fsycl)

target_compile_options(join_tests PRIVATE -fsycl)
target_link_options(join_tests PRIVATE -fsycl)

if(ENABLE_EXPERIMENTAL)
  target_compile_options(slab_tests PRIVATE -fsycl)
  target_link_options(slab_tests PRIVATE -fsycl)
endif()

add_test(scan_tests scan_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
add_test(hash_table_tests hash_table_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
add_test(cuckoo_hashtable_tests cuckoo_hashtable_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
add_test(join_tests join_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
if(ENABLE_EXPERIMENTAL)
  add_test(slab_tests slab_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
endif()
