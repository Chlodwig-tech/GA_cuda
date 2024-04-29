#ifndef BPOPULATION_CU
#define BPOPULATION_CU

#include "population.cu"

template<typename T, typename r> __global__ void BRandomKernel(T* *individuals, r a, r b){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    individuals[bid]->random(tid, a, b);
}

template<typename T, typename Type> __global__ void BCalculateFitnessKernel(T* *individuals, int size)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ Type shared_sum[];
    if(tid < size){
        shared_sum[tid] = individuals[bid]->fitnesspart(tid);
        __syncthreads();
        for(int i = 1; i < size; i *= 2){
            if(tid % (2 * i) == 0){
                shared_sum[tid] += shared_sum[tid + i];
            }
            __syncthreads();
        }
    }
    if(tid == 0){
        individuals[bid]->Set(shared_sum[0]);
    }
}

template<typename T> class BPopulation : public Population<T>{
    T helper;
public:
    BPopulation(int size) : Population<T>(size){}
    template<typename r> void random(r a, r b){
        BRandomKernel<<<this->size, 1024>>>(this->individuals, a, b);
    }
    void calculate_fitness(){
        auto f = helper.get_fvalue();
        int ssize = this->size * helper.get_fsize();
        BCalculateFitnessKernel<T, decltype(f)><<<this->size, 1024, ssize>>>(this->individuals, this->size);
    }
};

#endif // BPOPULATION_CU