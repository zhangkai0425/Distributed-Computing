const char *sgemm_desc = "Simple blocked sgemm.";
#include <immintrin.h>
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif
#define SMALL_BLOCK_SIZE 16
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

static void do_block_avx_k(int lda, int K, float *A, float *B, float *C)
{
    // 矩阵按列存储 A,B,C都是一维数组,lda是实际的矩阵大小(lda*lda),所以对矩阵元素来说 Aij = A[i + j * lda]
    // 下面是子块的16*16矩阵乘法,这里把传进来的A,B,C都看成16*16矩阵就行,比如传进来的A的首地址就是A分块之后子块的左上角首地址
    for (int j = 0; j < 16; ++j)
    {
        __m512 c = _mm512_loadu_ps(&C[lda * j]); // 循环16次，AVX-512指令会每次加载第j列中C中的16个float数 C = (c0,c1,...,c15)
        for (int k = 0; k < K; k++)
        {
            __m512 a = _mm512_loadu_ps(&A[lda * k]);   // 16次,每次加载第k列A中的16个float数
            __m512 b = _mm512_set1_ps(B[k + lda * j]); // 16次,对应每次加载第j列中第k个B中的数,并复制16次,成为元素相同的16位向量
            c = _mm512_fmadd_ps(a, b, c);              // 执行按位的向量乘法和加法 c = a * b + c 都是按位乘的
        }
        _mm512_storeu_ps(&C[lda * j], c); // 将c的值存到C的第j列当中
    }
}

static void do_block_avx_opt(int lda, float *A, float *B, float *C)
{
    __m512 c1 = _mm512_loadu_ps(&C[lda * 0]);
    __m512 c2 = _mm512_loadu_ps(&C[lda * 1]);
    __m512 c3 = _mm512_loadu_ps(&C[lda * 2]);
    __m512 c4 = _mm512_loadu_ps(&C[lda * 3]);
    __m512 c5 = _mm512_loadu_ps(&C[lda * 4]);
    __m512 c6 = _mm512_loadu_ps(&C[lda * 5]);
    __m512 c7 = _mm512_loadu_ps(&C[lda * 6]);
    __m512 c8 = _mm512_loadu_ps(&C[lda * 7]);
    __m512 c9 = _mm512_loadu_ps(&C[lda * 8]);
    __m512 c10 = _mm512_loadu_ps(&C[lda * 9]);
    __m512 c11 = _mm512_loadu_ps(&C[lda * 10]);
    __m512 c12 = _mm512_loadu_ps(&C[lda * 11]);
    __m512 c13 = _mm512_loadu_ps(&C[lda * 12]);
    __m512 c14 = _mm512_loadu_ps(&C[lda * 13]);
    __m512 c15 = _mm512_loadu_ps(&C[lda * 14]);
    __m512 c16 = _mm512_loadu_ps(&C[lda * 15]);
    for (int k = 0; k < 16; k += 4)
    {
        __m512 a1 = _mm512_loadu_ps(&A[lda * k]);
        __m512 a2 = _mm512_loadu_ps(&A[lda * (k + 1)]);
        __m512 a3 = _mm512_loadu_ps(&A[lda * (k + 2)]);
        __m512 a4 = _mm512_loadu_ps(&A[lda * (k + 3)]);

        __m512 b1_1 = _mm512_set1_ps(B[k + 0 + lda * 0]);
        __m512 b1_2 = _mm512_set1_ps(B[k + 1 + lda * 0]);
        __m512 b1_3 = _mm512_set1_ps(B[k + 2 + lda * 0]);
        __m512 b1_4 = _mm512_set1_ps(B[k + 3 + lda * 0]);

        __m512 b2_1 = _mm512_set1_ps(B[k + 0 + lda * 1]);
        __m512 b2_2 = _mm512_set1_ps(B[k + 1 + lda * 1]);
        __m512 b2_3 = _mm512_set1_ps(B[k + 2 + lda * 1]);
        __m512 b2_4 = _mm512_set1_ps(B[k + 3 + lda * 1]);

        __m512 b3_1 = _mm512_set1_ps(B[k + 0 + lda * 2]);
        __m512 b3_2 = _mm512_set1_ps(B[k + 1 + lda * 2]);
        __m512 b3_3 = _mm512_set1_ps(B[k + 2 + lda * 2]);
        __m512 b3_4 = _mm512_set1_ps(B[k + 3 + lda * 2]);

        __m512 b4_1 = _mm512_set1_ps(B[k + 0 + lda * 3]);
        __m512 b4_2 = _mm512_set1_ps(B[k + 1 + lda * 3]);
        __m512 b4_3 = _mm512_set1_ps(B[k + 2 + lda * 3]);
        __m512 b4_4 = _mm512_set1_ps(B[k + 3 + lda * 3]);

        __m512 b5_1 = _mm512_set1_ps(B[k + 0 + lda * 4]);
        __m512 b5_2 = _mm512_set1_ps(B[k + 1 + lda * 4]);
        __m512 b5_3 = _mm512_set1_ps(B[k + 2 + lda * 4]);
        __m512 b5_4 = _mm512_set1_ps(B[k + 3 + lda * 4]);

        __m512 b6_1 = _mm512_set1_ps(B[k + 0 + lda * 5]);
        __m512 b6_2 = _mm512_set1_ps(B[k + 1 + lda * 5]);
        __m512 b6_3 = _mm512_set1_ps(B[k + 2 + lda * 5]);
        __m512 b6_4 = _mm512_set1_ps(B[k + 3 + lda * 5]);

        __m512 b7_1 = _mm512_set1_ps(B[k + 0 + lda * 6]);
        __m512 b7_2 = _mm512_set1_ps(B[k + 1 + lda * 6]);
        __m512 b7_3 = _mm512_set1_ps(B[k + 2 + lda * 6]);
        __m512 b7_4 = _mm512_set1_ps(B[k + 3 + lda * 6]);

        __m512 b8_1 = _mm512_set1_ps(B[k + 0 + lda * 7]);
        __m512 b8_2 = _mm512_set1_ps(B[k + 1 + lda * 7]);
        __m512 b8_3 = _mm512_set1_ps(B[k + 2 + lda * 7]);
        __m512 b8_4 = _mm512_set1_ps(B[k + 3 + lda * 7]);

        __m512 b9_1 = _mm512_set1_ps(B[k + 0 + lda * 8]);
        __m512 b9_2 = _mm512_set1_ps(B[k + 1 + lda * 8]);
        __m512 b9_3 = _mm512_set1_ps(B[k + 2 + lda * 8]);
        __m512 b9_4 = _mm512_set1_ps(B[k + 3 + lda * 8]);

        __m512 b10_1 = _mm512_set1_ps(B[k + 0 + lda * 9]);
        __m512 b10_2 = _mm512_set1_ps(B[k + 1 + lda * 9]);
        __m512 b10_3 = _mm512_set1_ps(B[k + 2 + lda * 9]);
        __m512 b10_4 = _mm512_set1_ps(B[k + 3 + lda * 9]);

        __m512 b11_1 = _mm512_set1_ps(B[k + 0 + lda * 10]);
        __m512 b11_2 = _mm512_set1_ps(B[k + 1 + lda * 10]);
        __m512 b11_3 = _mm512_set1_ps(B[k + 2 + lda * 10]);
        __m512 b11_4 = _mm512_set1_ps(B[k + 3 + lda * 10]);

        __m512 b12_1 = _mm512_set1_ps(B[k + 0 + lda * 11]);
        __m512 b12_2 = _mm512_set1_ps(B[k + 1 + lda * 11]);
        __m512 b12_3 = _mm512_set1_ps(B[k + 2 + lda * 11]);
        __m512 b12_4 = _mm512_set1_ps(B[k + 3 + lda * 11]);

        __m512 b13_1 = _mm512_set1_ps(B[k + 0 + lda * 12]);
        __m512 b13_2 = _mm512_set1_ps(B[k + 1 + lda * 12]);
        __m512 b13_3 = _mm512_set1_ps(B[k + 2 + lda * 12]);
        __m512 b13_4 = _mm512_set1_ps(B[k + 3 + lda * 12]);

        __m512 b14_1 = _mm512_set1_ps(B[k + 0 + lda * 13]);
        __m512 b14_2 = _mm512_set1_ps(B[k + 1 + lda * 13]);
        __m512 b14_3 = _mm512_set1_ps(B[k + 2 + lda * 13]);
        __m512 b14_4 = _mm512_set1_ps(B[k + 3 + lda * 13]);

        __m512 b15_1 = _mm512_set1_ps(B[k + 0 + lda * 14]);
        __m512 b15_2 = _mm512_set1_ps(B[k + 1 + lda * 14]);
        __m512 b15_3 = _mm512_set1_ps(B[k + 2 + lda * 14]);
        __m512 b15_4 = _mm512_set1_ps(B[k + 3 + lda * 14]);

        __m512 b16_1 = _mm512_set1_ps(B[k + 0 + lda * 15]);
        __m512 b16_2 = _mm512_set1_ps(B[k + 1 + lda * 15]);
        __m512 b16_3 = _mm512_set1_ps(B[k + 2 + lda * 15]);
        __m512 b16_4 = _mm512_set1_ps(B[k + 3 + lda * 15]);

        c1 = _mm512_fmadd_ps(a1, b1_1, c1);
        c1 = _mm512_fmadd_ps(a2, b1_2, c1);
        c1 = _mm512_fmadd_ps(a3, b1_3, c1);
        c1 = _mm512_fmadd_ps(a4, b1_4, c1);

        c2 = _mm512_fmadd_ps(a1, b2_1, c2);
        c2 = _mm512_fmadd_ps(a2, b2_2, c2);
        c2 = _mm512_fmadd_ps(a3, b2_3, c2);
        c2 = _mm512_fmadd_ps(a4, b2_4, c2);

        c3 = _mm512_fmadd_ps(a1, b3_1, c3);
        c3 = _mm512_fmadd_ps(a2, b3_2, c3);
        c3 = _mm512_fmadd_ps(a3, b3_3, c3);
        c3 = _mm512_fmadd_ps(a4, b3_4, c3);

        c4 = _mm512_fmadd_ps(a1, b4_1, c4);
        c4 = _mm512_fmadd_ps(a2, b4_2, c4);
        c4 = _mm512_fmadd_ps(a3, b4_3, c4);
        c4 = _mm512_fmadd_ps(a4, b4_4, c4);

        c5 = _mm512_fmadd_ps(a1, b5_1, c5);
        c5 = _mm512_fmadd_ps(a2, b5_2, c5);
        c5 = _mm512_fmadd_ps(a3, b5_3, c5);
        c5 = _mm512_fmadd_ps(a4, b5_4, c5);

        c6 = _mm512_fmadd_ps(a1, b6_1, c6);
        c6 = _mm512_fmadd_ps(a2, b6_2, c6);
        c6 = _mm512_fmadd_ps(a3, b6_3, c6);
        c6 = _mm512_fmadd_ps(a4, b6_4, c6);

        c7 = _mm512_fmadd_ps(a1, b7_1, c7);
        c7 = _mm512_fmadd_ps(a2, b7_2, c7);
        c7 = _mm512_fmadd_ps(a3, b7_3, c7);
        c7 = _mm512_fmadd_ps(a4, b7_4, c7);

        c8 = _mm512_fmadd_ps(a1, b8_1, c8);
        c8 = _mm512_fmadd_ps(a2, b8_2, c8);
        c8 = _mm512_fmadd_ps(a3, b8_3, c8);
        c8 = _mm512_fmadd_ps(a4, b8_4, c8);

        c9 = _mm512_fmadd_ps(a1, b9_1, c9);
        c9 = _mm512_fmadd_ps(a2, b9_2, c9);
        c9 = _mm512_fmadd_ps(a3, b9_3, c9);
        c9 = _mm512_fmadd_ps(a4, b9_4, c9);

        c10 = _mm512_fmadd_ps(a1, b10_1, c10);
        c10 = _mm512_fmadd_ps(a2, b10_2, c10);
        c10 = _mm512_fmadd_ps(a3, b10_3, c10);
        c10 = _mm512_fmadd_ps(a4, b10_4, c10);

        c11 = _mm512_fmadd_ps(a1, b11_1, c11);
        c11 = _mm512_fmadd_ps(a2, b11_2, c11);
        c11 = _mm512_fmadd_ps(a3, b11_3, c11);
        c11 = _mm512_fmadd_ps(a4, b11_4, c11);

        c12 = _mm512_fmadd_ps(a1, b12_1, c12);
        c12 = _mm512_fmadd_ps(a2, b12_2, c12);
        c12 = _mm512_fmadd_ps(a3, b12_3, c12);
        c12 = _mm512_fmadd_ps(a4, b12_4, c12);

        c13 = _mm512_fmadd_ps(a1, b13_1, c13);
        c13 = _mm512_fmadd_ps(a2, b13_2, c13);
        c13 = _mm512_fmadd_ps(a3, b13_3, c13);
        c13 = _mm512_fmadd_ps(a4, b13_4, c13);

        c14 = _mm512_fmadd_ps(a1, b14_1, c14);
        c14 = _mm512_fmadd_ps(a2, b14_2, c14);
        c14 = _mm512_fmadd_ps(a3, b14_3, c14);
        c14 = _mm512_fmadd_ps(a4, b14_4, c14);

        c15 = _mm512_fmadd_ps(a1, b15_1, c15);
        c15 = _mm512_fmadd_ps(a2, b15_2, c15);
        c15 = _mm512_fmadd_ps(a3, b15_3, c15);
        c15 = _mm512_fmadd_ps(a4, b15_4, c15);

        c16 = _mm512_fmadd_ps(a1, b16_1, c16);
        c16 = _mm512_fmadd_ps(a2, b16_2, c16);
        c16 = _mm512_fmadd_ps(a3, b16_3, c16);
        c16 = _mm512_fmadd_ps(a4, b16_4, c16);
    }
    _mm512_storeu_ps(&C[lda * 0], c1);
    _mm512_storeu_ps(&C[lda * 1], c2);
    _mm512_storeu_ps(&C[lda * 2], c3);
    _mm512_storeu_ps(&C[lda * 3], c4);
    _mm512_storeu_ps(&C[lda * 4], c5);
    _mm512_storeu_ps(&C[lda * 5], c6);
    _mm512_storeu_ps(&C[lda * 6], c7);
    _mm512_storeu_ps(&C[lda * 7], c8);
    _mm512_storeu_ps(&C[lda * 8], c9);
    _mm512_storeu_ps(&C[lda * 9], c10);
    _mm512_storeu_ps(&C[lda * 10], c11);
    _mm512_storeu_ps(&C[lda * 11], c12);
    _mm512_storeu_ps(&C[lda * 12], c13);
    _mm512_storeu_ps(&C[lda * 13], c14);
    _mm512_storeu_ps(&C[lda * 14], c15);
    _mm512_storeu_ps(&C[lda * 15], c16);
}

static void do_block_large(int lda, int M, int N, int K, float *A, float *B, float *C)
{
    if ((lda < BLOCK_SIZE / 2 || lda < BLOCK_SIZE / 2 || lda < BLOCK_SIZE / 2) && (lda % SMALL_BLOCK_SIZE != 0))
        do_block(lda, M, N, K, A, B, C);
    else
        for (int i = 0; i < M; i += SMALL_BLOCK_SIZE)
            for (int j = 0; j < N; j += SMALL_BLOCK_SIZE)
                for (int k = 0; k < K; k += SMALL_BLOCK_SIZE)
                {
                    int M_part = min(SMALL_BLOCK_SIZE, M - i);
                    int N_part = min(SMALL_BLOCK_SIZE, N - j);
                    int K_part = min(SMALL_BLOCK_SIZE, K - k);
                    if (M_part == 16 && N_part == 16 && K_part == 16)
                        do_block_avx_opt(lda, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                    // do_block_avx_k(lda, K_part, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                    else
                        do_block_opt(lda, M_part, N_part, K_part, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                }
}

static void do_block_large_opt_align(int lda, int M, int N, int K, float *A, float *B, float *C)
{
    float AA[SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE];
    float BB[SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE];
    float CC[SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE];
    if ((lda < BLOCK_SIZE / 2 || lda < BLOCK_SIZE / 2 || lda < BLOCK_SIZE / 2) && (lda % SMALL_BLOCK_SIZE != 0) && 0)
        do_block(lda, M, N, K, A, B, C);
    else
        for (int i = 0; i < M; i += SMALL_BLOCK_SIZE)
            for (int j = 0; j < N; j += SMALL_BLOCK_SIZE)
                for (int k = 0; k < K; k += SMALL_BLOCK_SIZE)
                {
                    int M_part = min(SMALL_BLOCK_SIZE, M - i);
                    int N_part = min(SMALL_BLOCK_SIZE, N - j);
                    int K_part = min(SMALL_BLOCK_SIZE, K - k);
                    if (M_part == 16 && N_part == 16 && K_part == 16)
                        do_block_avx_opt(lda, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                    else
                    {
                        // pack A[M,K] to AA[16][16]
                        // pack B[K,N] to BB[16][16]
                        // pack C[M,N] to CC[16][16]
                        memset(AA, 0, sizeof(AA));
                        memset(BB, 0, sizeof(BB));
                        for (int x = 0; x < K_part; x++)
                            for (int y = 0; y < M_part; y++)
                                AA[y + x * SMALL_BLOCK_SIZE] = A[(y + i) + (x + k) * lda];
                        for (int x = 0; x < N_part; x++)
                            for (int y = 0; y < K_part; y++)
                                BB[y + x * SMALL_BLOCK_SIZE] = B[(y + k) + (x + j) * lda];
                        for (int x = 0; x < N_part; x++)
                            for (int y = 0; y < M_part; y++)
                                CC[y + x * SMALL_BLOCK_SIZE] = C[(y + i) + (x + j) * lda];
                        do_block_avx_opt(SMALL_BLOCK_SIZE, AA, BB, CC);
                        // unpack CC[16][16] to C[M,N]
                        for (int x = 0; x < N_part; x++)
                            for (int y = 0; y < M_part; y++)
                                C[(y + i) + (x + j) * lda] = CC[y + x * SMALL_BLOCK_SIZE];
                    }
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
