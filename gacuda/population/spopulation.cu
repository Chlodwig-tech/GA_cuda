#ifndef SPOPULATION_CU
#define SPOPULATION_CU

#include "population.cu"

template<typename T, typename r> __global__ void RandomKernel(T* *individuals, int size, r a, r b)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        individuals[tid]->random(tid, a, b);
    }
}
template<typename T> __global__ void CalculateFitnessKernel(T* *individuals, int size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        individuals[tid]->fitness();
    }
}


template<typename T> class SPopulation : public Population<T>{
public:
    SPopulation(int size) : Population<T>(size){}
    template<typename r> void random(r a, r b){
        RandomKernel<<<this->size / 1024 + 1, 1024>>>(this->individuals, this->size, a, b);
    }
    void calculate_fitness(){
        CalculateFitnessKernel<<<this->size / 1024 + 1, 1024>>>(this->individuals, this->size);
    }
};



#endif // SPOPULATION_CU