    #ifndef BINDIVIDUAL_CU
#define BINDIVIDUAL_CU

#include <curand_kernel.h>
#include "individual.cu"

template <typename T, typename Tfitness> __global__ void FitnessDynamicKernel(T *individual, int size){
    int tid = threadIdx.x;
    extern __shared__ Tfitness shared_sum[];
    shared_sum[tid] = individual->fitnesspart(tid);
    __syncthreads();
    for(int i = 1; i < size; i *= 2){
        if(tid % (2 * i) == 0){
            shared_sum[tid] += shared_sum[tid + i];
        }
        __syncthreads();
    }
    if(tid == 0){
        individual->Set(shared_sum[0]);
    }
}

template<typename DNA, typename Tfitness, int Size>
class BIndividual : public Individual<DNA, Tfitness, Size>{
public:
    __device__ virtual Tfitness fitnesspart(int index) = 0;
    __device__ virtual void fitness(){
        //FitnessDynamicKernel<BIndividual<DNA, Tfitness, Size>, Tfitness><<<1, Size, Size * sizeof(Tfitness)>>>(this, Size);
        int sum = 0;
        for(int i = 0; i < Size; i++){
            sum += this->fitnesspart(i);
        }
        this->fvalue = sum;
    }

    __device__ void random(int tid, DNA a, DNA b);
};

template<typename DNA, typename Tfitness, int Size>
__device__ void BIndividual<DNA, Tfitness, Size>::random(int tid, DNA a, DNA b){
    if(tid < Size){
        printf("xd\n");
        curandState state;
        curand_init(clock64(), tid, 0, &state);
        this->genes[tid] = generateRandomNumber(&state, a, b);
    }
}

#endif // BINDIVIDUAL_CU