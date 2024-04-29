#include <iostream>
#include <curand_kernel.h>

#include "gacuda/gacuda.h"

#include <random>


__device__ int *costs;

template<int Size> class SOrganism : public SIndividual<int, int, Size>{
public:    
    __device__ virtual void fitness(){
        int sum = 0;
        for(int i = 0; i < Size - 1; i++){
            sum += costs[this->genes[i] * Size + this->genes[i + 1]];
        }
        sum += costs[this->genes[Size - 1] * Size + this->genes[0]];
        this->fvalue = sum;
    }
    __device__ void crossover(
    curandState *state, SOrganism<Size> *secondParent)
    {
        int part = curand_uniform(state) * Size;
        SOrganism<Size> *child = new SOrganism<Size>();
        memcpy(child->genes, this->genes, part * sizeof(int));

        for(int i = 0; i < Size; i++){
            int gene = secondParent->genes[i];
            bool used = false;
            for(int j = 0; j < part && !used; j++){
                if(child->genes[j] == gene){
                    used = true;
                }
            }
            if(!used){
                child->genes[part] = gene;
                part++;
            }
        }

        child->fitness();

        if(this->fvalue < secondParent->fvalue){
            if(secondParent->fvalue > child->fvalue){
                memcpy(secondParent->genes, this->genes, Size * sizeof(int));
            }
        }else{
            if(this->fvalue > child->fvalue){
                memcpy(this->genes, secondParent->genes, Size * sizeof(int));
            }
        }
        delete child;
    }


    __device__ virtual void print(){
        for(int i = 0; i < Size; i++){
            printf("%d ", this->genes[i]);
        }
        printf("-> %d\n", this->fvalue);
    }

    __device__ void random(int tid, int a, int b){
        curandState state;
        curand_init(clock64(), tid, 0, &state);
        for(int i = 0; i < Size; i++){
            this->genes[i] = i;
        }
        for(int i = Size - 1; i > 0; i--){
            int x = curand_uniform(&state) * 100;
            int j = x % (i + 1);
            int temp = this->genes[i];
            this->genes[i] = this->genes[j];
            this->genes[j] = temp;
        }
    }
    
};


int main(){
    const int isize = 8;
    const int psize = 1024;

	std::random_device rd;
    std::mt19937 g(rd());
	std::uniform_int_distribution<int> distribution(1, isize);
    int* prices = (int*)malloc(sizeof(int) * isize * isize);
	for(int i = 0; i < isize; i++){
		for(int j = i; j < isize; j++){
			if(i == j){
				prices[i * isize + j] = 0;
			}else{
				int value = distribution(g);
				prices[i * isize + j] = value;
				prices[j * isize + i] = value;
			}
		}
	}

    for(int i = 0; i < isize; i++){
		for(int j = i; j < isize; j++){
			printf("%d ", prices[i * isize + j]);
		}
        printf("\n");
	}

    int *ah;
    cudaMalloc((void **)&ah, psize * isize * sizeof(int));
    cudaMemcpy(ah, prices, psize * isize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(costs, &ah, sizeof(int *), size_t(0),cudaMemcpyHostToDevice);

    SPopulation<SOrganism<isize>> p(psize);
    p.random(1, isize);
    p.calculate_fitness();
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