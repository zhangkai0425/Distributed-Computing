const char *sgemm_desc = "Simple blocked sgemm.";
#include <immintrin.h>
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, float *A, float *B, float *C)
{
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
        {
            register float b = B[k + j * lda];
            /* For each row i of A */
            for (int i = 0; i < M; ++i)
            {
                C[i + j * lda] += A[i + k * lda] * b;
            }
        }
}

static void do_block_pack(int lda, int M, int N, int K, float *A, float *B, float *C)
{
    // pack B to continuous memory in cache
    float BB[BLOCK_SIZE * BLOCK_SIZE];
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
            BB[k + j * BLOCK_SIZE] = B[k + j * lda];
    for (int k = 0; k < K; ++k)
        /* For each column j of B */
        for (int j = 0; j < N; ++j)
        {
            register float b = BB[k + j * BLOCK_SIZE];
            /* For each row i of A */
            for (int i = 0; i < M; ++i)
            {
                /* Compute C(i,j) */
                C[i + j * lda] += A[i + k * lda] * b;
            }
        }
}

static void do_block_avx(int lda, float *A, float *B, float *C)
{
    // 矩阵按列存储 A,B,C都是一维数组,lda是实际的矩阵大小(lda*lda),所以对矩阵元素来说 Aij = A[i + j * lda]
    // 下面是子块的16*16矩阵乘法,这里把传进来的A,B,C都看成16*16矩阵就行,比如传进来的A的首地址就是A分块之后子块的左上角首地址
    for (int j = 0; j < 16; ++j)
    {
        __m512 c = _mm512_loadu_ps(&C[lda * j]); // 循环16次，AVX-512指令会每次加载第j列中C中的16个float数 C = (c0,c1,...,c15)
        for (int k = 0; k < 16; k++)
        {
            __m512 a = _mm512_loadu_ps(&A[lda * k]);   // 16次,每次加载第k列A中的16个float数
            __m512 b = _mm512_set1_ps(B[k + lda * j]); // 16次,对应每次加载第j列中第k个B中的数,并复制16次,成为元素相同的16位向量
            c = _mm512_fmadd_ps(a, b, c);              // 执行按位的向量乘法和加法 c = a * b + c 都是按位乘的
        }
        _mm512_storeu_ps(&C[lda * j], c); // 将c的值存到C的第j列当中
    }
}

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_sgemm(int lda, float *__restrict__ A, float *__restrict__ B, float *__restrict__ C)
{
    /* For each block-row of A */
    for (int i = 0; i < lda; i += BLOCK_SIZE)
        /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE)
            /* Accumulate block sgemms into block of C */
            for (int k = 0; k < lda; k += BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);

                /* Perform individual block sgemm */
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
}
