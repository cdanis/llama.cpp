# vendored third-party libraries shared by multiple programs

add_subdirectory(cpp-httplib)

# hash libraries, shared by llama-gguf-hash and llama-logits-hash

add_library(xxhash OBJECT xxhash/xxhash.c xxhash/xxhash.h)
add_library(sha1 OBJECT sha1/sha1.c sha1/sha1.h)
add_library(sha256 OBJECT sha256/sha256.c sha256/sha256.h)
if (NOT MSVC)
    # disable warnings in 3rd party code
    target_compile_options(sha1 PRIVATE -w)
    target_compile_options(sha256 PRIVATE -w)
endif()

# sha256.c includes "rotate-bits/rotate-bits.h"
target_include_directories(sha256 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
