struct instance_params;
struct h_instance_params;

#include <pthread.h>

typedef pthread_t CUTThread;
typedef void *(*CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC void
#define  CUT_THREADEND

//Create thread
CUTThread start_thread(CUT_THREADROUTINE func, void * data){
    pthread_t thread;
    pthread_create(&thread, NULL, func, data);
    return thread;
}

//Wait for thread to finish
void end_thread(CUTThread thread){
    pthread_join(thread, NULL);
}

//Destroy thread
void destroy_thread( CUTThread thread ){
    pthread_cancel(thread);
}

//Wait for multiple threads
void wait_for_threads(const CUTThread * threads, int num){
    for(int i = 0; i < num; i++)
        end_thread( threads[i] );
}



#include <stdio.h>
#include <math.h>
#include <string.h>

#include "../depends/libstl-cuda/memory.cuh"
#include "../depends/libstl-cuda/vector.cuh"
#include "../depends/libstl-cuda/utility.cuh"

#include "../depends/libff-cuda/fields/bigint_host.cuh"
#include "../depends/libff-cuda/fields/fp_host.cuh"
#include "../depends/libff-cuda/fields/fp2_host.cuh"
#include "../depends/libff-cuda/fields/fp6_3over2_host.cuh"
#include "../depends/libff-cuda/fields/fp12_2over3over2_host.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_init_host.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_g1_host.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_g2_host.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_pp_host.cuh"
#include "../depends/libmatrix-cuda/transpose/transpose_ell2csr.cuh"
#include "../depends/libmatrix-cuda/spmv/csr-balanced.cuh"
#include "../depends/libff-cuda/scalar_multiplication/multiexp.cuh"


#include "../depends/libff-cuda/curves/bls12_381/bls12_381_init.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_pp.cuh"

#include <time.h>
#include <set>

using namespace libff;

struct instance_params
{
    bls12_381_Fr instance;
    bls12_381_G1 g1_instance;
    bls12_381_G2 g2_instance;
    bls12_381_GT gt_instance;
};

struct h_instance_params
{
    bls12_381_Fr_host h_instance;
    bls12_381_G1_host h_g1_instance;
    bls12_381_G2_host h_g2_instance;
    bls12_381_GT_host h_gt_instance;
};


template<typename ppT>
struct MSM_params
{
    libstl::vector<libff::Fr<ppT>> vf;
    libstl::vector<libff::G1<ppT>> vg;
};


__global__ void init_params()
{
    gmp_init_allocator_();
    bls12_381_pp::init_public_params();
}

__global__ void instance_init(instance_params* ip)
{
    ip->instance = bls12_381_Fr(&bls12_381_fp_params_r);
    ip->g1_instance = bls12_381_G1(&g1_params);
    ip->g2_instance = bls12_381_G2(&g2_params);
    ip->gt_instance = bls12_381_GT(&bls12_381_fp12_params_q);
}

void instance_init_host(h_instance_params* ip)
{
    ip->h_instance = bls12_381_Fr_host(&bls12_381_fp_params_r_host);
    ip->h_g1_instance = bls12_381_G1_host(&g1_params_host);
    ip->h_g2_instance = bls12_381_G2_host(&g2_params_host);
    ip->h_gt_instance = bls12_381_GT_host(&bls12_381_fp12_params_q_host);
}


template<typename ppT>
__global__ void generate_MP(MSM_params<ppT>* mp, instance_params* ip, size_t size)
{
    new ((void*)mp) MSM_params<ppT>();
    mp->vf.presize(size, 512, 32);
    mp->vg.presize(size, 512, 32);

    libstl::launch<<<512, 32>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            libff::Fr<ppT> f = ip->instance.random_element();
            libff::G1<ppT> g = ip->g1_instance.random_element();
            f ^= idx;
            g = g * idx;
            while(idx < size)
            {
                mp->vf[idx] = f;
                mp->vg[idx] = g;
                f = f + f;
                g = g + g;
                idx += tnum;
            }
        }
    );
    cudaDeviceSynchronize();

    ip->g1_instance.p_batch_to_special(mp->vg, 160, 32);
}

struct Mem
{
    size_t device_id;
    void* mem;
};

void* multi_init_params(void* params)
{
    Mem* device_mem = (Mem*) params;
    cudaSetDevice(device_mem->device_id);
    size_t init_size = 1024 * 1024 * 1024;
    init_size *= 2;
    if( cudaMalloc( (void**)&device_mem->mem, init_size ) != cudaSuccess) printf("device malloc error!\n");
    libstl::initAllocator(device_mem->mem, init_size);
    init_params<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}

struct Instance
{
    size_t device_id;
    instance_params** ip;
};

void* multi_instance_init(void* instance)
{
    Instance* it = (Instance*)instance;
    cudaSetDevice(it->device_id);
    if( cudaMalloc( (void**)it->ip, sizeof(instance_params)) != cudaSuccess) printf("ip malloc error!\n");
    instance_init<<<1, 1>>>(*it->ip);
    cudaDeviceSynchronize();
    return 0;
}

template<typename ppT>
struct MSM
{
    size_t device_id;
    MSM_params<ppT>* mp;
    instance_params* ip;
    libff::G1<ppT>* res;
};

template<typename ppT>
void* multi_MSM(void* msm)
{
    MSM<ppT>* it = (MSM<ppT>*)msm;
    cudaSetDevice(it->device_id);

    size_t lockMem;
    libstl::lock_host(lockMem);
    libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(it->mp->vg, it->mp->vf, it->ip->instance, it->ip->g1_instance, 512, 32);
    cudaDeviceSynchronize();
    libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(it->mp->vg, it->mp->vf, it->ip->instance, it->ip->g1_instance, 512, 32);
    cudaDeviceSynchronize();
    libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(it->mp->vg, it->mp->vf, it->ip->instance, it->ip->g1_instance, 512, 32);
    cudaDeviceSynchronize();
    libstl::resetlock_host(lockMem);

    cudaEvent_t eventMSMStart, eventMSMEnd;
    cudaEventCreate( &eventMSMStart);
	cudaEventCreate( &eventMSMEnd);
    cudaEventRecord( eventMSMStart, 0); 
    cudaEventSynchronize(eventMSMStart);

    for(size_t i=0; i<1; i++)
    {
        it->res = libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(it->mp->vg, it->mp->vf, it->ip->instance, it->ip->g1_instance, 512, 32);
        cudaDeviceSynchronize();
    }

    cudaEventRecord( eventMSMEnd, 0);
    cudaEventSynchronize(eventMSMEnd);
    float   TimeMSM;
    cudaEventElapsedTime( &TimeMSM, eventMSMStart, eventMSMEnd );
    printf( "Time thread %lu for MSM:  %3.5f ms\n", it->device_id, TimeMSM );

    return 0;
}

template<typename ppT_host, typename ppT_device>
void D2H(libff::G1<ppT_host>* hg1, libff::G1<ppT_device>* dg1, libff::G1<ppT_host>* g1_instance)
{
    cudaMemcpy(hg1, dg1, sizeof(libff::G1<ppT_device>), cudaMemcpyDeviceToHost);
    hg1->set_params(g1_instance->params);
}


template<typename ppT>
void Reduce(libff::G1<ppT>* hg1, libff::Fr<ppT>* instance, size_t total)
{
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    libff::G1<ppT> g1 = hg1[device_count-1];

    if(device_count != 1)
    {
        for(size_t i=device_count - 2; i <= device_count - 1; i--)
        {
            size_t log2_total = libff::log2(total);
            size_t c = log2_total - (log2_total / 3 - 2);
            size_t num_bits = instance->size_in_bits();
            size_t num_groups = (num_bits + c - 1) / c;
            size_t sgroup = (num_groups + device_count - 1) / device_count * i;
            size_t egroup = (num_groups + device_count - 1) / device_count * (i + 1);
            if(egroup > num_groups) egroup = num_groups;
            if(sgroup > num_groups) sgroup = num_groups;
            if(egroup == sgroup) continue;

            for(size_t j=0; j < (egroup - sgroup) * c; j++)
            {
                g1 = g1.dbl();
            }
            g1 = g1 + hg1[i];
        }
    }

    g1.to_special();

}

__device__ long omega2( long n){
    long rem = n % 2;
    long exponent = 0;
    while( rem == 0){
        exponent ++;
        n >>= 1;
        rem = n % 2;
    }
    return exponent;
}

__device__ long omega3( long n){
    long rem = n % 3;
    long exponent = 0;
    while( rem == 0){
        exponent ++;
        n = n/3;
        rem = n % 3;
    }
    return exponent;
}
#include "../depends/libstl-cuda/set.cu"
__device__ int8_t *bucket;
__global__ void construct_B0_bucket(long start, long limit)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // get the thread index
    if(idx >= start && idx <= limit) {
        //printf("Processing %lu\n", idx);
        if (((omega2(idx) + omega3(idx))%2) == 0){
            bucket[idx] = 1;
        }
    }
}
__global__ void subtract_B1_omega2(long q, long start, long end)
{
    // size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // get the thread index
    // size_t nidx = idx+start;
    // if(nidx >= start && nidx < end) {
    //     if (bucket[nidx] && bucket[q - 2*nidx] ) { // if both are in the bucket
    //         bucket[q - 2*nidx] = 0; // remove q-2*idx from the bucket
    //     }
    // }
    for(long nidx = start; nidx < end; nidx++)
    {
        if (bucket[nidx] && bucket[q - 2*nidx]) { // if both are in the bucket
            bucket[q - 2*nidx] = 0; // remove q-2*idx from the bucket
            // printf("Removing %lu from bucket\n", q - 2*nidx);
        }
    }
}
__global__ void subtract_B1_omega3(long q, long start, long end)
{
    // size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // get the thread index
    // size_t nidx = idx+start;
    // if(nidx >= start && nidx < end) {
    //     if (bucket[nidx] && bucket[q - 3*nidx] ) { // if both are in the bucket
    //         bucket[q - 3*nidx] = 0; // remove q-3*idx from the bucket
    //     }
    // }
    for(long nidx = start; nidx < end; nidx++)
    {
        if (bucket[nidx] && bucket[q - 3*nidx]) { // if both are in the bucket
            bucket[q - 3*nidx] = 0; // remove q-3*idx from the bucket
            // printf("Removing %lu from bucket\n", q - 3*nidx);
        }
    }
}
__global__ void construct_bucket_set(size_t size, libstl::set<long> *b_set)
{
    b_set = new libstl::set<long>(size, 0);
    size_t count = 0;
    for(int i=0; i< size; i++)
    {
        if(bucket[i] == 1) {
            b_set->insert(i); // insert into the set
        }
    }
    printf("Bucket size: %lu\n", b_set->size()); // print the size of the bucket set
}
__global__ void construct_bucket(const long q, const long ah)
{
    bucket[0] = 1;
    bucket[1] = 1;

    int blocksize = 128;
    int gridsize = ((q+2) + blocksize - 1) / blocksize;

    construct_B0_bucket<<<gridsize, blocksize>>>(2,(long)(q/2)); // construct the B0 bucket for 1 to q/2
    
    // gridsize = ((q/2 - q/4 + 2)+ blocksize -1)/blocksize;
    subtract_B1_omega2<<<1, 1>>>(q, (long)(q/4), (long)(q/2)); // subtract B1 omega2 from q/2 to q/4

    //gridsize = ((q/4 - q/6 + 2)+ blocksize -1)/blocksize;
    subtract_B1_omega3<<<1, 1>>>(q, (long)(q/6), (long)(q/4)); // subtract B1 omega3 from q/4 to q/6
    gridsize = ((ah+2)+ blocksize -1)/blocksize;
    construct_B0_bucket<<<gridsize, blocksize>>>(1,(long)(ah+1));
    
}
struct mbflag
{
    int m;
    long b;
    bool flag;
    __host__ __device__ mbflag() : m(0), b(0), flag(false) {}
    __host__ __device__ mbflag(int _m, long _b, bool _flag) : m(_m), b(_b), flag(_flag) {}
};
__device__ libstl::vector<mbflag> *decompose;

__global__ void decompose_builder1(libstl::set<long> *b_set, long q, long bsize)
{
    if (decompose == NULL) return;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // get the thread index
    
    if(idx <bsize)
    {
        int b = (*b_set)[idx];
        for(int m=1; m<=3; m++) {
            if (m*b <= q) {
                (*decompose)[q - m*b] = *(new mbflag(m,b,1));
            }
        }
    }
}
__global__ void decompose_builder2(libstl::set<long> *b_set, long q, long bsize)
{
    // this is a helper function to build the decomposition map
    if (decompose == NULL) return;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // get the thread index

    if(idx < bsize)
    {
        int b = (*b_set)[idx];
        for(int m=1; m<=3; m++) {
            if (m*b <= q) {
                (*decompose)[m*b] = *(new mbflag(m,b,0)); // set the flag to 0 for the direct multiples
            }
        }
    }
}

__global__ void construct_decomposition_map(long q, libstl::set<long> *b_set)
{
    int MULTI_SET[] = {1, 2, 3}; // the multi set for decomposition, can be extended

    decompose = new libstl::vector<mbflag>(q+2); // create a new set on device
    size_t bsize = b_set->size(); // get the size of the b_set
    size_t blocksize = 128; // block size for the kernel launch
    size_t gridsize = ((bsize) + blocksize - 1) / blocksize; // calculate the grid size

    decompose_builder1<<<gridsize, blocksize>>>(b_set, q, bsize); // build the first pass of decomposition map
    decompose_builder2<<<gridsize, blocksize>>>(b_set, q, bsize); // build the second pass of decomposition map
    cudaDeviceSynchronize(); // ensure the kernel has finished executing
    printf("Came here?\n");
}

__device__ libstl::vector<libff::G1<bls12_381_pp>> *precomputed_points;
__device__ void precompute_points(libstl::vector<libff::G1<bls12_381_pp>> *vg, libstl::vector<libff::G1<bls12_381_pp>> *precomputed_points)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tnum = blockDim.x * gridDim.x;
    (*precomputed_points)[3*idx] = (*vg)[idx];
    (*precomputed_points)[3*idx+1] = (*vg)[idx].dbl();
    (*precomputed_points)[3*idx+2] = (*vg)[idx] + (*precomputed_points)[3*idx+1];
}
__global__ void precompute(libstl::vector<libff::G1<bls12_381_pp>> *vg, libstl::vector<libff::G1<bls12_381_pp>> *precomputed_points)
{
    precomputed_points = new libstl::vector<libff::G1<bls12_381_pp>>();
}
__global__ void test_set()
{
    libstl::set<long> s(10); 
    assert(s.size() == 0); // Initially empty
    assert(!s.contains(5)); // Should not contain 5
    s.insert(5); // Insert 5
    assert(s.contains(5)); // Now it should contain 5
    assert(s.size() == 1); // Size should be 1
    s.insert(10); // Insert 10
    assert(s.contains(10)); // Now it should contain 10
    assert(s.size() == 2); // Size should be 2
    s.insert(5); // Insert duplicate 5, should not change size
    assert(s.contains(5)); // Still contains 5
    assert(s.size() == 2); // Size should still be 2
    s.remove(5); // Remove 5
    assert(!s.contains(5)); // Now it should not contain 5
    assert(s.size() == 1); // Size should be 1 after removing 5
    s.remove(10); // Remove 10
    assert(!s.contains(10)); // Now it should not contain 10
    assert(s.size() == 0); // Size should be 0 after removing all elements  
    s.remove(20); // Removing a non-existent element should not change the size
    assert(s.size() == 0); // Size should still be 0
}
__device__ libstl::set<long> *b_set;
int main(int argc, char* argv[])
{
    if (argc < 2) {
		printf("Please enter the MSM scales (e.g. 20 represents 2^20) \n");
		return 1;
	}
    //precompute<<<1, 1>>>(NULL, NULL);
    int log_size = atoi(argv[1]);

    int deviceCount;
    cudaGetDeviceCount( &deviceCount );
    CUTThread  thread[deviceCount];

    bls12_381_pp_host::init_public_params();
    cudaSetDevice(0);

    size_t num_v = (size_t) (1 << log_size);

    // params init 
    Mem device_mem[deviceCount];
    for(size_t i=0; i<deviceCount; i++)
    {
        device_mem[i].device_id = i;
        device_mem[i].mem = NULL;
        thread[i] = start_thread( multi_init_params, &device_mem[i] );
    }
    for(size_t i=0; i<deviceCount; i++)
    {
        end_thread(thread[i]);
    }

    // instance init
    instance_params* ip[deviceCount];
    Instance instance[deviceCount];
    for(size_t i=0; i<deviceCount; i++)
    {
        instance[i].device_id = i;
        instance[i].ip = &ip[i];
        thread[i] = start_thread( multi_instance_init, &instance[i] );
    }
    for(size_t i=0; i<deviceCount; i++)
    {
        end_thread(thread[i]);
    }

    h_instance_params hip;
    instance_init_host(&hip);

    // elements generation
    MSM_params<bls12_381_pp>* mp[deviceCount];
    for(size_t i=0; i<deviceCount; i++)
    {
        cudaSetDevice(i);
        if( cudaMalloc( (void**)&mp[i], sizeof(MSM_params<bls12_381_pp>)) != cudaSuccess) printf("mp malloc error!\n");
        generate_MP<bls12_381_pp><<<1, 1>>>(mp[i], ip[i], num_v);
    }
    for(size_t i=0; i<deviceCount; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    cudaSetDevice(0);

    printf("MSM Scalar: %lu\n", mp[0]->vf.size_host());
    printf("MSM Point: %lu\n", mp[0]->vg.size_host());


    // generate bucket 
    //ip[0]->instance.params->modulus->print(); // print the modulus for debugging
    
    // mp[0]->vf[0].params->modulus->print();

    // bigint<4> *r = ip[0]->instance.params->modulus;
    // ip[0]->instance.params->modulus->num_bits();
    // printf("438\n");
    size_t *num_bits = new size_t();//ip[0]->instance.params->num_bits;
    // libstl::launch<<<1, 1>>>
    // (
    //     [=]
    //     __device__ (libff::bigint<4> *n)
    //     {
    //         n->print();
    //         *num_bits = n->num_bits(); // get the number of bits in the modulus

    //     }, ip[0]->instance.params->modulus
    // );
    size_t total = mp[0]->vf.size_host();
    size_t log2_total = log2(total);
    size_t s = log2_total - (log2_total / 3 - 2);
    size_t q = (1 << s);
    size_t rhBitLength = *num_bits%s;
    size_t limbCount = *num_bits/s;
    size_t *ah = new size_t(65);
    libstl::launch<<<1, 1>>>
    (
        [=]
        __device__ (size_t *ah)
        {
            bucket = new int8_t[(1<<(s-1))+2];
            memset(bucket, 0, sizeof(int8_t)*((1<<(s-1))+2)); 
            // for(int i=0;i<rhBitLength;i++)
            // {
            //     // if(r.test_bit(limbCount*s + i))
            //     // {
            //     //     *ah = *ah | (1 << (i)); 
            //     // }
            // }
        }, ah
    );
    cudaDeviceSynchronize(); // ensure the kernel has finished executing

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    construct_bucket<<<1, 1, 0, stream>>>(q, *ah); // construct the bucket on device

    construct_bucket_set<<<1,1, 0, stream>>> ((q/2)+2, b_set);
    construct_decomposition_map<<<1,1,0, stream>>>(q, b_set); // construct the decomposition map
    cudaStreamSynchronize(stream); // ensure the kernel has finished executing
    cudaStreamDestroy(stream); // destroy the stream

    cudaStream_t stream2;
    cudaStreamCreate(&stream2);
    libstl::launch<<<1, 1, 0, stream2>>>
    (
        [=]
        __device__ ()
        {
            for(int i=0;i<100; i++)
            {
                printf("%d ", i);
            }
        }
    );
    cudaStreamSynchronize(stream2); // ensure the kernel has finished executing
    cudaStreamDestroy(stream2); // destroy the stream
    cudaDeviceSynchronize();

    // precompute points
    size_t grid = (num_v + 31 )/32;
    size_t block = 32;
    
    // (*precomputed_points).presize(num_v*3, grid*3, block);
    // __device__ libstl::vector<libff::G1<bls12_381_pp>> points = mp[0]->vg;
    
    test_set<<<1,1>>>(); 
    // perform pippenger msm

    // printf("Precomputed Size: %lu\n", (*precomputed_points).size_host());


    // msm
    // MSM<bls12_381_pp> msm[deviceCount];
    // for(size_t i=0; i<deviceCount; i++)
    // {
    //     msm[i].device_id = i;
    //     msm[i].mp = mp[i];
    //     msm[i].ip = ip[i];
    //     thread[i] = start_thread( multi_MSM<bls12_381_pp>, &msm[i] );
    // }
    // for(size_t i=0; i<deviceCount; i++)
    // {
    //     end_thread(thread[i]);
    // }

    // libff::G1<bls12_381_pp_host> hg1[deviceCount];
    // for(size_t i=0; i < deviceCount; i++)
    // {
    //     cudaSetDevice(i);
    //     D2H<bls12_381_pp_host, bls12_381_pp>(&hg1[i], msm[i].res, &hip.h_g1_instance);
    // }

    // Reduce<bls12_381_pp_host>(hg1, &hip.h_instance, num_v);

    cudaDeviceReset();
    return 0;
}
