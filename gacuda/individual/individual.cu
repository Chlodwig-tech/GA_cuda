#ifndef INDIVIDUAL_CU
#define INDIVIDUAL_CU

#include <curand_kernel.h>

template<typename DNA, typename Tfitness, int Size>
class Individual{
protected:    
    DNA genes[Size];
    Tfitness fvalue;
public:
    __device__ virtual void random(int tid, DNA a, DNA b) = 0;
    __device__ virtual void fitness() = 0;
    __device__ virtual void print(){}

    __device__ bool is_greater(Individual *other);
    __device__ void mutate(curandState *state);
    template<typename TIndividual> __device__ void crossoverwithchild(
        curandState *state, TIndividual *secondParent, TIndividual *child);
    template<typename TIndividual> __device__ void crossover(
        curandState *state, TIndividual *secondParent);
    __device__ void Set(Individual *other);
    __device__ void Set(Tfitness fval);
    
    __host__ int get_fsize();
    __host__ Tfitness get_fvalue();
};


template<typename DNA, typename Tfitness, int Size>
__host__ Tfitness Individual<DNA, Tfitness, Size>::get_fvalue(){
    return fvalue;
}

template<typename DNA, typename Tfitness, int Size>
__host__ int Individual<DNA, Tfitness, Size>::get_fsize(){
    return sizeof(fvalue);
}

template<typename DNA, typename Tfitness, int Size>
__device__ void Individual<DNA, Tfitness, Size>::Set(Tfitness fval){
    this->fvalue = fval;
}

template<typename DNA, typename Tfitness, int Size>
__device__ void Individual<DNA, Tfitness, Size>::Set(Individual *other){
    memcpy(this->genes, other->genes, Size * sizeof(DNA));
    this->fvalue = other->fvalue;
}

template<typename DNA, typename Tfitness, int Size>
template<typename TIndividual>
__device__ void Individual<DNA, Tfitness, Size>::crossover(
    curandState *state, TIndividual *secondParent)
{
    int part = curand_uniform(state) * Size;
    TIndividual *child = new TIndividual();
    memcpy(child->genes, this->genes, part * sizeof(DNA));
    memcpy(child->genes + part, secondParent->genes + part, (Size - part) * sizeof(DNA));
    child->fitness();
    if(this->fvalue < secondParent->fvalue){
        if(secondParent->fvalue > child->fvalue){
            memcpy(secondParent->genes, this->genes, part * sizeof(DNA));
        }
    }else{
        if(this->fvalue > child->fvalue){
            memcpy(this->genes + part, secondParent->genes + part, (Size - part) * sizeof(DNA));
        }
    }
    delete child;
}

template<typename DNA, typename Tfitness, int Size>
template<typename TIndividual>
__device__ void Individual<DNA, Tfitness, Size>::crossoverwithchild(
        curandState *state, TIndividual *secondParent, TIndividual *child)
{
    int part = curand_uniform(state) * Size;
    memcpy(child->genes, this->genes, part * sizeof(DNA));
    memcpy(child->genes + part, secondParent->genes + part, (Size - part) * sizeof(DNA));
    child->fitness();
}

template<typename DNA, typename Tfitness, int Size>
__device__ void Individual<DNA, Tfitness, Size>::mutate(curandState *state){
    int i = curand_uniform(state) * Size;
    int j = curand_uniform(state) * Size;
    DNA temp = genes[i];
    genes[i] = genes[j];
    genes[j] = temp;
}

template<typename DNA, typename Tfitness, int Size>
__device__ bool Individual<DNA, Tfitness, Size>::is_greater(Individual *other){
    return this->fvalue > other->fvalue;
}

template<typename T> __device__ T generateRandomNumber(curandState *state, T a, T b){
    return a + curand_uniform(state) * (b - a);
}

#endif // INDIVIDUAL_CU