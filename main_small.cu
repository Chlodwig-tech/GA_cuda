#include <iostream>

#include "gacuda/gacuda.h"

template<int Size> class Organism : public SIndividual<int, int, Size>{
public:
    __device__ virtual void fitness(){
        int sum = 0;
        for(int i = 0; i < Size; i++){
            sum += this->genes[i];
        }
        this->fvalue = sum;
    }
    __device__ virtual void print(){
        for(int i = 0; i < Size; i++){
            printf("%d ", this->genes[i]);
        }
        printf("-> %d\n", this->fvalue);
    }
};

int main(){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int isize = 8;
    const int psize = 8;

    SPopulation<Organism<isize>> p(psize);

    p.random(0, 10);
    p.calculate_fitness();
    p.sort();
    p.print();
    cudaDeviceSynchronize();
    printf("------------------------------------------\n");

    int number_of_epochs = 10000;
    cudaEventRecord(start);
    /*for(int i = 0; i < number_of_epochs; i++){
        p.mutate(10.0f);
        p.crossover(10.0f);
        p.sort();
    }*/
    p.run(number_of_epochs, 10.0f, 10.0f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    p.print();

    cudaDeviceSynchronize();
    printf("-------------------------------\n");
    p.print_best();
    cudaDeviceSynchronize();
    printf("Time taken: %f\n", milliseconds);
}