// C API for reinforcement learning environments

// Environment libraries are implemented as normal C shared libraries, providing
// the interface described here.  Each library must implement all functions.
//
// Since many languages can export C shared libraries, this lets you make your
// RL environment in almost any language you want, without the overhead of
// managing a socket connection.

#pragma once

// allow this header to be included by a C++ file
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#if defined(__GNUC__)
#define LIBENV_API __attribute__((__dllexport__))
#else
#define LIBENV_API __declspec(dllexport)
#endif
#else
#if defined(__GNUC__)
#define LIBENV_API __attribute__((__visibility__("default")))
#else
#define LIBENV_API
#endif
#endif

// BEGIN_CDEF

#define LIBENV_MAX_NAME_LEN 128
#define LIBENV_MAX_NDIM 16
#define LIBENV_VERSION 1  // also appears in libenv.py

// opaque type for the environment returned by the library
    typedef void libenv_env;

    // data types used by scalar_type
    enum libenv_dtype {
        LIBENV_DTYPE_UNUSED = 0,
        LIBENV_DTYPE_UINT8 = 1,
        LIBENV_DTYPE_INT32 = 2,
        LIBENV_DTYPE_FLOAT32 = 3,
    };

    // an instance of one of the above data types, use the field corresponding
    // to the dtype that is being used
    union libenv_value {
        uint8_t uint8;
        int32_t int32;
        float float32;
    };

    // scalar types used as eltype for tensors
    enum libenv_scalar_type {
        LIBENV_SCALAR_TYPE_UNUSED = 0,
        LIBENV_SCALAR_TYPE_REAL = 1,
        LIBENV_SCALAR_TYPE_DISCRETE = 2,
    };

    // different spaces used by the environments
    // for a libenv environment, a space is a collection of (name, tensortype) pairs
    // so the top level space is always a DictType
    enum libenv_space_name {
        LIBENV_SPACE_UNUSED = 0,
        LIBENV_SPACE_OBSERVATION = 1,
        LIBENV_SPACE_ACTION = 2,
        LIBENV_SPACE_INFO = 3,
    };

    // libenv_tensortype describes the fixed-size data used to communicate with an
    // environment
    //
    // each individual tensortype corresponds to a fixed sized contiguous array of data
    // equivalent to a numpy array
    struct libenv_tensortype {
        char name[LIBENV_MAX_NAME_LEN];
        enum libenv_scalar_type scalar_type;
        enum libenv_dtype dtype;
        int shape[LIBENV_MAX_NDIM];
        int ndim;
        union libenv_value low;
        union libenv_value high;
    };

    // libenv_option holds a name-data pair used to configure environment instances
    //
    // the buffer pointed to by the data pointer will be kept alive by python for the
    // duration of the env
    struct libenv_option {
        char name[LIBENV_MAX_NAME_LEN];
        enum libenv_dtype dtype;
        int count;
        void* data;
    };

    // libenv_options holds an array of libenv_option instances
    struct libenv_options {
        struct libenv_option* items;
        int count;
    };

    // libenv_buffers contains pointers to buffers used by libenv_observe() and libenv_act()
    // all memory will be allocated by the caller of libenv_set_buffers() (normally libenv.py)
    // and must remain valid until the environment is destroyed with libenv_close()
    //
    // ob and infos are arrays of buffer pointers, the length is determined
    // by the length of the corresponding spaces object times the number of environments
    //
    // the spaces are arranged in a structure of arrays fashion, for instance,
    // given an observation space with keys, (uint8) shapes of {'a': [1], 'b': [1]},
    // and num_envs = 2, the memory layout looks like this:
    //  0                   1
    //  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
    // +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    // |    env 1 a    |    env 2 a    |
    // +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    //  0                   1
    //  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
    // +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    // |    env 1 b    |    env 2 b    |
    // +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    //
    struct libenv_buffers {
        void** ob;  // array of pointers with length len(ob_space) * num_envs
        float* rew;  // pointer to num_envs floats
        uint8_t* first;  // pointer to num_envs uint8s
        void** info;  // array of pointers with length len(info_space) * num_envs
        void** ac;  // array of pointers with length len(ac_space) * num_envs
    };

#if !defined(NO_PROTOTYPE)

    // libenv_version returns the version of libenv that was used to compile the environment
    // in order to detect old versions and provide an error to the user
    //
    // the implementation of this function body should just be "return LIBENV_VERSION;"
    LIBENV_API int libenv_version();

    // libenv_make creates a new environment instance
    //
    // returns a pointer to an opaque environment instance
    // the instance being pointed to belongs to the library, use libenv_close() to destroy it
    //
    // environments are not required to be thread-safe, but different environments
    // must be usable from different threads simultaneously
    //
    // num_envs is the number of environments to create, these may actually be multiple agents
    // in the same environment, but that is up to the individual environment to decide
    //
    // options is a series of key-value pairs whose meaning is only known to an individual environment library
    LIBENV_API libenv_env* libenv_make(int num, const struct libenv_options options);

    // libenv_get_tensortypes gets a description of the spaces used by this environment instance
    // the caller must allocate the correct number of spaces and owns the memory
    // if called with a null pointer for spaces, returns the number of spaces that are required
    LIBENV_API int libenv_get_tensortypes(libenv_env* handle, enum libenv_space_name name, struct libenv_tensortype* types);

    // libenv_set_buffers provides a libenv_buffers object with pointers to where the environment should write data
    //
    // the bufs object belongs to the caller, along with all buffers pointed to by that object
    // which must be allocated by the caller
    LIBENV_API void libenv_set_buffers(libenv_env* handle, struct libenv_buffers* bufs);

    // libenv_observe observes the environment, along with the reward, first flag, and the info values
    //
    // if the episode is done, the `ob` value of the observation will be from the new episode
    LIBENV_API void libenv_observe(libenv_env* handle);

    // libenv_act submits an action to the environment that has already been put into the acts buffer
    // provided with libenv_set_buffers()
    //
    // the environment may apply the action immediately, in a thread, or else just store it and apply it when
    // libenv_observe() is called.  the action must remain valid for the duration of the call to libenv_act()
    // but no guarantees are made outside of that.
    //
    // the step object belongs to the caller, along with the ob buffer which must be allocated by the caller
    // the step object must stay valid until step_wait is called
    LIBENV_API void libenv_act(libenv_env* handle);

    // libenv_close closes the environment and frees any resources associated with it
    LIBENV_API void libenv_close(libenv_env* handle);

#endif

// END_CDEF

#ifdef __cplusplus
}
#endif