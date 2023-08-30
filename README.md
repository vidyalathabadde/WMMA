# `cudaTensorCoreGemm` Sample

This sample demonstrates a GEMM computation using the unified matrix interface (joint_matrix APIs) introduced in SYCL 2020 revision 5 specifications.
This interface is intended to unify different tensor hardware: Intel AMX in CPUs, Intel XMX in Intel GPUs, Habana Gaudi and Goya tensor and gemm cores, Nvidia TPUs, IBM Power MMA. All these hardware provide low-level intrinsics or assembly to access and perform matrix operations.

| Area              | Description
|:---                   |:---
| What you will learn              | Migrate cudaTensorCoreGemm from CUDA to SYCL
| Time to complete              | 15 minutes
| Category                      | Concepts and Functionality

## Purpose

The sample shows the migration of cudaTensorCoreGemm from CUDA to SYCL using SYCLomatic tool using the unified matrix interface (joint_matrix APIs) that unifies Intel AMX, Intel XMX, and Nvidia Tensor Cores. The goal is to provide a unified interface that is portable but also benefits from the maximum performance these different hardware can offer.

>**Note**: We use Intel® open-sources SYCLomatic migration tool which assists developers in porting CUDA code automatically to SYCL code. To finish the process, developers complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. Users can also use Intel® DPC++ Compatibility Tool which comes along with the Intel® oneAPI Base Toolkit.

This sample contains two versions in the following folders:

| Folder Name                   | Description
|:---                           |:---
| `01_dpct_output`              | Contains the output of SYCLomatic Tool which is a partially migrated version of CUDA code.
| `02_sycl_migrated`            | Contains the manually migrated sycl code

## Workflow For CUDA to SYCL migration

Refer [Workflow](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for details.

## CUDA source code evaluation

This sample is migrated from the NVIDIA CUDA sample. See the sample [cudaTensorCoreGemm](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/cudaTensorCoreGemm) in the NVIDIA/cuda-samples GitHub.

## Prerequisites

| Optimized for              | Description
|:---                   |:---
| OS                    | Ubuntu* 22.04
| Hardware              | Intel® Data Center GPU Max
| Software                | SYCLomatic (Tag - 20230720) <br> Intel® oneAPI Base Toolkit version 2023.2.1

For more information on how to install Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.v354cy).

## Key Implementation Details

This sample demonstrates the migration of the following CUDA features: 

- Cuda WMMA APIs
- Shared memory
- Element-wise operations in CUDA

## Build the `cudaTensorCoreGemm` Sample for AMX and XMX hardware

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### Tool assisted migration – SYCLomatic 

For this sample, the SYCLomatic tool automatically migrates 100% of the CUDA runtime API's to SYCL. Follow these steps to generate the SYCL code using the compatibility tool:

1. git clone https://github.com/NVIDIA/cuda-samples.git
2. cd cuda-samples/Samples/3_CUDA_Features/cudaTensorCoreGemm
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
4. The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.
5. Pass the JSON file as input to the SYCLomatic Tool. The result is written to a folder named dpct_output. The --in-root specifies path to the root of the source tree to be migrated.
   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function --use-experimental-features=matrix
   ```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
