#include <iostream>
#include <curand_kernel.h>

#include "gacuda/gacuda.h"

template<int Size> class BOrganism : public BIndividual<int, int, Size>{
public:    
    __device__ virtual int fitnesspart(int index){
        return -this->genes[index];
    }
    __device__ virtual void print(){
        for(int i = 0; i < Size; i++){
            printf("%d ", this->genes[i]);
        }
        printf("-> %d\n", this->fvalue);
    }
};

template<int Size> class SOrganism : public SIndividual<int, int, Size>{
public:    
    __device__ virtual void fitness(){
        char target[] = "Jaufdbusfubasiubduaisdasdsdfuishfsuihfaiusfhauishf";
        int sum = 0;
        for(int i = 0; i < Size; i++){
            if(this->genes[i] != target[i]){
                sum++;
            }
        }
        this->fvalue = sum;
    }
    __device__ virtual void print(){
        for(int i = 0; i < Size; i++){
            printf("%c", this->genes[i]);
        }
        printf("-> %d\n", this->fvalue);
    }
    __device__ void random(int tid, int a, int b){
        curandState state;
        curand_init(clock64(), tid, 0, &state);
        for(int i = 0; i < Size; i++){
            int x = curand_uniform(&state) * 100;
            this->genes[i] = (x % 58) + 65;
        }
    }
};


int main(){
    const int isize = sizeof("Jaufdbusfubasiubduaisdasdsdfuishfsuihfaiusfhauishf") - 1;
    //const int isize = 8;
    const int psize = 1024;

    SPopulation<SOrganism<isize>> p(psize);
    //BPopulation<SOrganism<isize>> p(psize);
    p.random(0, 10);
    p.print(2);
    cudaDeviceSynchronize();
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