#ifndef SINDIVIDUAL_CU
#define SINDIVIDUAL_CU

#include <curand_kernel.h>
#include "individual.cu"

template<typename DNA, typename Tfitness, int Size>
class SIndividual : public Individual<DNA, Tfitness, Size>{
public:
    __device__ void random(int tid, DNA a, DNA b);
};

template<typename DNA, typename Tfitness, int Size>
__device__ void SIndividual<DNA, Tfitness, Size>::random(int tid, DNA a, DNA b){
    //printf("xd");
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    for(int i = 0; i < Size; i++){
        this->genes[i] = generateRandomNumber(&state, a, b);
    }
}

#endif // SINDIVIDUAL_CU