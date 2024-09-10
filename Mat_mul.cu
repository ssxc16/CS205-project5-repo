#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <cblas.h>
#include <cmath>
#define BLOCK_SIZE 32

#define CHECK_MATRIX_VALID(matrix1, matrix2, M, N, K, ret_val) \
    do { \
        if ((matrix1) == NULL) { \
            fprintf(stderr, "Invalid matrix, matrix1 or its data is NULL\n"); \
            return ret_val; \
        } \
        if ((matrix2) == NULL) { \
            fprintf(stderr, "Invalid matrix, matrix2 or its data is NULL\n"); \
            return ret_val; \
        } \
        if ((M) <= 0 || (N) <= 0 || (K) <= 0) { \
            fprintf(stderr, "Invalid matrix, M, N or K is less than 0\n"); \
            return ret_val; \
        } \
    } while (0)

__global__ void matrixMul(float* __restrict__ d_A, float* __restrict__ d_B, float* d_C, int n) {
    __shared__ float2 s_a[512];
    __shared__ float2 s_b[512];
    // Thread result matrix
    float c[64] = {0.f};
    // Thread registers
    float2 a[4];
    float2 b[4];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = threadIdx.x;
    // Continuous warp access
    int tidx = x % 32;
    int tidy = x / 32;
    int v = x % 16;
    int u = x / 16;
    

    d_A += (by * 128 + tidx) + tidy * n;
    d_B += tidy * n + bx * 128 + tidx;
    d_C += (by * 128 + u * 2) * n + bx * 128 + v * 2;

    for (int i = 0; i < n; i += 8) {
        ((float*) s_a)[tidy * 128 + tidx] = d_A[0];
        ((float*) s_a)[tidy * 128 + tidx + 32] = d_A[32];
        ((float*) s_a)[tidy * 128 + tidx + 64] = d_A[64];
        ((float*) s_a)[tidy * 128 + tidx + 96] = d_A[96];

        ((float*) s_b)[tidy * 128 + tidx] = d_B[0];
        ((float*) s_b)[tidy * 128 + tidx + 32] = d_B[32];
        ((float*) s_b)[tidy * 128 + tidx + 64] = d_B[64];
        ((float*) s_b)[tidy * 128 + tidx + 96] = d_B[96];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < 8; k++) {
            // load relational s_a,s_b data into a,b registers
            {
                a[0] = s_a[k * 64 + 0 + u];
                a[1] = s_a[k * 64 + 16 + u];
                a[2] = s_a[k * 64 + 32 + u];
                a[3] = s_a[k * 64 + 48 + u];
                b[0] = s_b[k * 64 + 0 + v];
                b[1] = s_b[k * 64 + 16 + v];
                b[2] = s_b[k * 64 + 32 + v];
                b[3] = s_b[k * 64 + 48 + v];
            }

            // use a,b to calculate c 
            {
                c[0] += a[0].x * b[0].x;
                c[1] += a[0].y * b[0].x;
                c[2] += a[1].x * b[0].x;
                c[3] += a[1].y * b[0].x;
                c[4] += a[2].x * b[0].x;
                c[5] += a[2].y * b[0].x;
                c[6] += a[3].x * b[0].x;
                c[7] += a[3].y * b[0].x;
                c[8] += a[0].x * b[0].y;
                c[9] += a[0].y * b[0].y;
                c[10] += a[1].x * b[0].y;
                c[11] += a[1].y * b[0].y;
                c[12] += a[2].x * b[0].y;
                c[13] += a[2].y * b[0].y;
                c[14] += a[3].x * b[0].y;
                c[15] += a[3].y * b[0].y;
                c[16] += a[0].x * b[1].x;
                c[17] += a[0].y * b[1].x;
                c[18] += a[1].x * b[1].x;
                c[19] += a[1].y * b[1].x;
                c[20] += a[2].x * b[1].x;
                c[21] += a[2].y * b[1].x;
                c[22] += a[3].x * b[1].x;
                c[23] += a[3].y * b[1].x;
                c[24] += a[0].x * b[1].y;
                c[25] += a[0].y * b[1].y;
                c[26] += a[1].x * b[1].y;
                c[27] += a[1].y * b[1].y;
                c[28] += a[2].x * b[1].y;
                c[29] += a[2].y * b[1].y;
                c[30] += a[3].x * b[1].y;
                c[31] += a[3].y * b[1].y;
                c[32] += a[0].x * b[2].x;
                c[33] += a[0].y * b[2].x;
                c[34] += a[1].x * b[2].x;
                c[35] += a[1].y * b[2].x;
                c[36] += a[2].x * b[2].x;
                c[37] += a[2].y * b[2].x;
                c[38] += a[3].x * b[2].x;
                c[39] += a[3].y * b[2].x;
                c[40] += a[0].x * b[2].y;
                c[41] += a[0].y * b[2].y;
                c[42] += a[1].x * b[2].y;
                c[43] += a[1].y * b[2].y;
                c[44] += a[2].x * b[2].y;
                c[45] += a[2].y * b[2].y;
                c[46] += a[3].x * b[2].y;
                c[47] += a[3].y * b[2].y;
                c[48] += a[0].x * b[3].x;
                c[49] += a[0].y * b[3].x;
                c[50] += a[1].x * b[3].x;
                c[51] += a[1].y * b[3].x;
                c[52] += a[2].x * b[3].x;
                c[53] += a[2].y * b[3].x;
                c[54] += a[3].x * b[3].x;
                c[55] += a[3].y * b[3].x;
                c[56] += a[0].x * b[3].y;
                c[57] += a[0].y * b[3].y;
                c[58] += a[1].x * b[3].y;
                c[59] += a[1].y * b[3].y;
                c[60] += a[2].x * b[3].y;
                c[61] += a[2].y * b[3].y;
                c[62] += a[3].x * b[3].y;
                c[63] += a[3].y * b[3].y;
            }
        }

        __syncthreads();
        d_A += 8 * n;
        d_B += 8 * n;
    }
    // write the final value to global memory
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        d_C[0] = c[j];
        d_C[1] = c[j + 8];
        d_C[32] = c[j + 16];
        d_C[33] = c[j + 24];
        d_C[64] = c[j + 32];
        d_C[65] = c[j + 40];
        d_C[96] = c[j + 48];
        d_C[97] = c[j + 56];
        d_C += n;
        d_C[0] = c[j + 1];
        d_C[1] = c[j + 9];
        d_C[32] = c[j + 17];
        d_C[33] = c[j + 25];
        d_C[64] = c[j + 33];
        d_C[65] = c[j + 41];
        d_C[96] = c[j + 49];
        d_C[97] = c[j + 57];
        d_C += 31 * n;
    }
}

__global__ void scaleAddKernel(float* A, float* B, float a, float b, int n) {
  
    int idx = blockIdx.x * blockDim.x + threadIdx.x % 32;
    int idy = blockIdx.y * blockDim.y + threadIdx.x / 32;
    int index = idy * n + idx;
    float temp = A[index];

    B[index] = a * temp + b;
}

__global__ void matrixMulNaive(float *A, float *B, float *C, size_t Width) {

    size_t bx = blockIdx.x, by = blockIdx.y;
    size_t tx = threadIdx.x, ty = threadIdx.y;
    size_t Row = by * BLOCK_SIZE + ty;
    size_t Col = bx * BLOCK_SIZE + tx;

    float temp = 0;
    for (size_t j = 0; j < Width; ++j) {
        temp += A[Row*Width + j] * B[Col * Width + j];
    }
    C[Row*Width + Col] = temp;
}

__host__ void matrixT(float* A, int N) {
    float temp;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            temp = A[i * N + j];
            A[i * N + j] = A[j * N + i];
            A[j * N + i] = temp;
        }
    }
}

__host__ void sum(float* A, int N) {
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%f ", A[i * N + j]);
    //     }
    //     printf("\n");
    // }
    // printf("A[10] is %f\n", A[10]);
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        float* temp = A + i * N;
        for (int j = 0; j < N; j++) {
            sum += temp[j];
        }
    }
    printf("sum is %f\n", sum);
}

__host__ void countDiff(float* A, float* B, int N) {
    // float a1 = A[5 * N + 1];
    // float b1 = B[5 * N + 1];
    // float a2 = A[1 * N + 5];
    // float b2 = B[1 * N + 5];
    // printf("A[5][1] is %f, B[5][1] is %f\n", a1, b1);
    // printf("A[1][5] is %f, B[1][5] is %f\n", a2, b2);
    double Diff = 0.0;
    for (int i = 0; i < N; i++) {
        float* temp1 = A + i * N;
        float* temp2 = B + i * N;
        for (int j = 0; j < N; j++) {
            Diff += abs(temp1[j] - temp2[j]);
            // printf("Diff is %f\n", abs(temp1[j] - temp2[j]));
        }
    }
    printf("Diff is %f\n", Diff);
}

__host__ void initMatrix(float* matrixA, float* matrixB, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // if (i == 2*j){
            //     matrixA[i * N + j] = 1.0;
            //     matrixB[i * N + j] = 1.0;
            // }
            // else{
            //     matrixA[i * N + j] = 0.0;
            //     matrixB[i * N + j] = 0.0;
            // }

            // matrixA[i * N + j] = j % 3 ? 1.0 : 0.5;
            // matrixB[i * N + j] = j % 3 ? 1.0 : 0.5;
            // matrixA[i * N + j] = 1;
            // matrixB[i * N + j] = 1;
            // matrixA[i * N + j] = (float) rand() / RAND_MAX;
            // matrixB[i * N + j] = (float) rand() / RAND_MAX;
            
            // matrixA[i * N + j] = fmod((float)1.01 * i, 3);
            // matrixB[i * N + j] = fmod((float)1.1 * j, 3);
            matrixA[i * N + j] = i*2 + j;
            matrixB[i * N + j] = i*2 + j;
        }
    }
}

int main() {
    float *matrixA, *matrixB, *myResult, *cuResult, *cResult, *scaleResult;
    float *dev_A, *dev_B, *dev_C;
    size_t N = 16384;

    float milliseconds;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    matrixA = (float*)aligned_alloc(32, N * N * sizeof(float));
    matrixB = (float*)aligned_alloc(32, N * N * sizeof(float));
    myResult = (float*)aligned_alloc(32, N * N * sizeof(float));
    cuResult = (float*)aligned_alloc(32, N * N * sizeof(float));
    cResult = (float*)aligned_alloc(32, N * N * sizeof(float));
    scaleResult = (float*)aligned_alloc(32, N * N * sizeof(float));

    cudaMalloc(&dev_A, N * N * sizeof(float));
    cudaMalloc(&dev_B, N * N * sizeof(float));
    cudaMalloc(&dev_C, N * N * sizeof(float));

    initMatrix(matrixA, matrixB, N);

    CHECK_MATRIX_VALID(matrixA, matrixB, N, N, N, -1);
    matrixT(matrixA, N);
    cudaMemcpy(dev_A, matrixA, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, matrixB, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32 * 32);
    dim3 gridDim((N + 32 - 1) / 32, (N + 32 - 1) / 32);
    cudaEventRecord(start);
    scaleAddKernel<<<gridDim, blockDim>>>(dev_A, dev_C, 1.0, 0.0, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("The scaleAddKernel time used is %fms\n", milliseconds);
    cudaMemcpy(scaleResult, dev_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    dim3 blocks(N / 128, N / 128);
    dim3 threads(256);
    cudaEventRecord(start);
    matrixMul<<<blocks, threads>>>(dev_A, dev_B, dev_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(myResult, dev_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Mine time used is %fms\n", milliseconds);


    cublasHandle_t handle;
    cublasCreate(&handle);
    float a2 = 1, b2 = 0;
    cudaEventRecord(start);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &a2, dev_A, N, dev_B, N, &b2, dev_C, N);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(cuResult, dev_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("The cuBLAS time used is %fms\n", milliseconds);


    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, matrixA, N, matrixB, N, 0.0, cResult, N);
    matrixT(myResult, N);
    sum(myResult,N);
    sum(cuResult, N);
    sum(cResult, N);
    countDiff(cResult,cuResult,N);
    
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    free(matrixA);
    free(matrixB);
    free(myResult);

    return 0;
}