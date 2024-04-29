#ifndef POPULATION_CU
#define POPULATION_CU

#define CUDA_CALL(x, message) {if((x) != cudaSuccess) { \
    printf("Error - %s(%d)[%s]: %s\n", __FILE__, __LINE__, message, cudaGetErrorString(x)); \
    exit(EXIT_FAILURE); }}

template<typename T> __global__ void initIndividualsKernel(T* *individuals, int size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        individuals[tid] = new T();
    }
}
template<typename T> __global__ void deinitIndividualsKernel(T* *individuals, int size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        delete individuals[tid];
    }
}
template<typename T> __global__ void printKernel(T* *individuals, int size, int max)
{
    for(int i = 0; i < size && i < max; i++){
        individuals[i]->print();
    }
    printf("\n");
}
template<typename T> __global__ void PerformMutations(T* *individuals, int size, float probability)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < size){
        curandState state;
        curand_init(clock64(), index, 0, &state);
        if(curand_uniform(&state) * 100.0f < probability){
            individuals[index]->mutate(&state);
            individuals[index]->fitness();
        }
    }
}
template<typename T> __global__ void PerformCrossover(T* *individuals, int size, float probability)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < size){
        curandState state;
        curand_init(clock64(), index, 0, &state);
        bool cond = curand_uniform(&state) * 100.0f < probability;
        if(cond){
            int parent_index = curand_uniform(&state) * size;
            individuals[index]->crossover(&state, individuals[parent_index]);
            individuals[index]->fitness();
        }
    }
}
template<typename T> __global__ void bitonicSortKernel(T* *individuals, int size)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(int k = 2; k <= size; k <<= 1){
        for(int j = k >> 1; j > 0; j >>= 1){
            int ixj = tid ^ j;
            if(tid < size && ixj < size && ixj > tid && 
                !(((tid & k) == 0) ^ (individuals[tid]->is_greater(individuals[ixj])))
            )
            {
                T* temp = individuals[tid];
                individuals[tid] = individuals[ixj];
                individuals[ixj] = temp;
            }
            __syncthreads();
        }
    }   
}
template<typename T> __global__ void runKernel(
    T* *individuals, T* *best, int size, float mprobability, float pprobability, int number_of_epochs
)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        for(int i = 0; i < number_of_epochs; i++){
            curandState state;
            curand_init(clock64(), tid, 0, &state);
            if(curand_uniform(&state) * 100.0f < mprobability){
                individuals[tid]->mutate(&state);
                individuals[tid]->fitness();
            }
            if(curand_uniform(&state) * 100.0f < pprobability){
                int parent_tid = curand_uniform(&state) * size;
                individuals[tid]->crossover(&state, individuals[parent_tid]);
                individuals[tid]->fitness();
            }
            for(int k = 2; k <= size; k <<= 1){
                for(int j = k >> 1; j > 0; j >>= 1){
                    int ixj = tid ^ j;
                    if(tid < size && ixj < size && ixj > tid && 
                        !(((tid & k) == 0) ^ (individuals[tid]->is_greater(individuals[ixj])))
                    )
                    {
                        T* temp = individuals[tid];
                        individuals[tid] = individuals[ixj];
                        individuals[ixj] = temp;
                    }
                    __syncthreads();
                }
            }
            if(best[0]->is_greater(individuals[0])){
                best[0]->Set(individuals[0]);
            }
        }
    }
}

template<typename T> class Population{
protected:
    T* *individuals;
    T*  *best;
    int size;
public:
    Population(int size) : size(size){
        CUDA_CALL(cudaMalloc((void **)&individuals, size * sizeof(T)), "Population cudaMalloc");
        initIndividualsKernel<<<size / 1024 + 1, 1024>>>(individuals, size);
        CUDA_CALL(cudaMalloc((void **)&best, size * sizeof(T)), "Population best cudaMalloc");
        initIndividualsKernel<<<1, 1>>>(best, 1);
    }
    ~Population(){
        deinitIndividualsKernel<<<size / 1024 + 1, 1024>>>(individuals, size);
        CUDA_CALL(cudaFree(individuals), "Population cudaFree");
        deinitIndividualsKernel<<<1, 1>>>(best, 1);
        CUDA_CALL(cudaFree(best), "Population best cudaFree");
    }
    void print(int max=-1){
        printKernel<<<1, 1>>>(individuals, size, max == -1 ? size:max);
        cudaDeviceSynchronize();
        printf("\n");
    }
    void print_best(){
        printKernel<<<1, 1>>>(best, 1, 1);
    }
    void sort(){
        bitonicSortKernel<<<size / 1024 + 1, 1024>>>(individuals, size);
    }
    void mutate(float probability=1.0f){
        PerformMutations<<<size / 1024 + 1, 1024>>>(individuals, size, probability);
    }
    void crossover(float probability=1.0f){
        PerformCrossover<<<size / 1024 + 1, 1024>>>(individuals, size, probability);
    }
    void run(int number_of_epochs, float mprobability=1.0f, float pprobability=1.0f){
        runKernel<<<this->size / 1024 + 1, 1024>>>(this->individuals, this->best, this->size, mprobability, pprobability, number_of_epochs);
    }
};

#endif // POPULATION_CU