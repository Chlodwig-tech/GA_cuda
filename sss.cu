#include <iostream>
#include <curand_kernel.h>

#include "gacuda/gacuda.h"

template<int Size> class Organism : public SIndividual<int, int, Size>{
public:
    __device__ virtual void fitness(){
        int sum = 0;
        for(int i = 0; i < Size; i++){
            sum += this->genes[i];
        }
        this->fvalue = -sum;
    }
    __device__ virtual void print(){
        for(int i = 0; i < Size; i++){
            printf("%d ", this->genes[i]);
        }
        printf("-> %d\n", this->fvalue);
    }
};

int main(){
    const int isize = 8;
    const int psize = 1024;

    SPopulation<Organism<isize>> p(psize);

    p.random(0, 10);
    p.calculate_fitness();
    p.sort();
    p.print(2);
    cudaDeviceSynchronize();

    int number_of_epochs = 1000;
    for(int i = 0; i < number_of_epochs; i++){
        p.mutate(10.0f);
        p.crossover(10.0f);
        p.sort();
    }

    p.print(2);
}