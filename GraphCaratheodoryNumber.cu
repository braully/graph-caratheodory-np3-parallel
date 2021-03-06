#include "UndirectedSparseGraph.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define DEFAULT_THREAD_PER_BLOCK 64
#define MAX_DEFAULT_SIZE_QUEUE 128
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))



//References and Examples:
//https://msdn.microsoft.com/en-us/library/aa289166(v=vs.71).aspx
//D. Knuth. The Art of Computer Programming: Generating All Combinations and Partitions. Number v. 3-4 in Art of Computer Programming. Addison-Wesley, 2005.


#define verboseSerial true
#define verboseSerialDetailed false
#define verboseParallel true
#define verboseParallelDetailed false

__host__ __device__
long maxCombinations(long n, long k) {
    if (n == 0 || k == 0) {
        return 0;
    }
    if (n < k) {
        return 0;
    }
    if (n == k) {
        return 1;
    }
    long delta, idxMax;
    if (k < n - k) {
        delta = n - k;
        idxMax = k;
    } else {
        delta = k;
        idxMax = n - k;
    }

    long ans = delta + 1;
    for (long i = 2; i <= idxMax; ++i) {
        ans = (ans * (delta + i)) / i;
    }
    return ans;
}

__host__ __device__
long maxCombinations(long n, long k, unsigned long *cacheMaxCombination) {
    if (cacheMaxCombination[k]) {
        return cacheMaxCombination[k];
    }
    return cacheMaxCombination[k] = maxCombinations(n, k);
}

__host__ __device__
void initialCombination(long n, long k, long* combinationArray, long idx, unsigned long *cacheMaxCombination) {
    long a = n;
    long b = k;
    long x = (maxCombinations(n, k) - 1) - idx;
    for (long i = 0; i < k; ++i) {
        combinationArray[i] = a - 1;
        while (maxCombinations(combinationArray[i], b) > x) {
            --combinationArray[i];
        }
        x = x - maxCombinations(combinationArray[i], b);
        a = combinationArray[i];
        b = b - 1;
    }

    for (long i = 0; i < k; ++i) {
        combinationArray[i] = (n - 1) - combinationArray[i];
    }
}

__host__ __device__
void initialCombination(long n, long k, long* combinationArray, long idx) {
    long a = n;
    long b = k;
    long x = (maxCombinations(n, k) - 1) - idx;
    for (long i = 0; i < k; ++i) {
        combinationArray[i] = a - 1;
        while (maxCombinations(combinationArray[i], b) > x) {
            --combinationArray[i];
        }
        x = x - maxCombinations(combinationArray[i], b);
        a = combinationArray[i];
        b = b - 1;
    }

    for (long i = 0; i < k; ++i) {
        combinationArray[i] = (n - 1) - combinationArray[i];
    }
}

__host__ __device__
void initialCombination(long n, long k, long* combinationArray) {
    for (long i = 0; i < k; i++) {
        combinationArray[i] = i;
    }
}

__host__ __device__
void nextCombination(long n,
        long k,
        long* currentCombination) {
    if (currentCombination[0] == n - k) {
        return;
    }
    long i;
    for (i = k - 1; i > 0 && currentCombination[i] == n - k + i; --i);
    ++currentCombination[i];
    for (long j = i; j < k - 1; ++j) {
        currentCombination[j + 1] = currentCombination[j] + 1;
    }
}

__host__ __device__ void printCombination(long *currentCombination,
        long sizeComb) {
    printf("S = {");
    for (long i = 0; i < sizeComb; i++) {
        printf("%2d", currentCombination[i]);
        if (i < sizeComb - 1) {
            printf(", ");
        }
    }
    printf(" }");
}

__host__ __device__
void prlongQueue(long *queue, long headQueue, long tailQueue) {
    printf("Queue(%d):{", tailQueue - headQueue);
    for (long i = headQueue; i <= tailQueue; i++) {
        printf("%2d", queue[i]);
        if (i < tailQueue) {
            printf(", ");
        }
    }
    printf(" }\n");
}

__host__ __device__
long calcDerivatedPartial(long *csrColIdxs, long nvertices,
        long *csrRowOffset, long sizeRowOffset,
        unsigned char *aux, unsigned char *auxc,
        long auxSize, long *currentCombination,
        long sizeComb, long *queue, long maxSizeQueue) {


    for (long i = 0; i < sizeComb; i++) {
        long p = currentCombination[i];
        long headQueue = 0;
        long tailQueue = -1;

        for (long j = 0; j < auxSize; j++) {
            auxc[j] = 0;
        }

        for (long j = 0; j < sizeComb; j++) {
            long v = currentCombination[j];
            if (v != p) {
                tailQueue = (tailQueue + 1) % maxSizeQueue;
                queue[tailQueue] = v;
                auxc[v] = INCLUDED;
            }
        }
        while (headQueue <= tailQueue) {
            long verti = queue[headQueue];
            headQueue = (headQueue + 1) % maxSizeQueue;
            aux[verti] = 0;
            if (verti < nvertices) {
                long end = csrColIdxs[verti + 1];
                for (long x = csrColIdxs[verti]; x < end; x++) {
                    long vertn = csrRowOffset[x];
                    if (vertn != verti) {
                        int previousValue = auxc[vertn];
                        auxc[vertn] = auxc[vertn] + NEIGHBOOR_COUNT_INCLUDED;
                        if (previousValue < INCLUDED && auxc[vertn] >= INCLUDED) {
                            tailQueue = (tailQueue + 1) % maxSizeQueue;
                            queue[tailQueue] = vertn;
                        }
                    }
                }
            }
        }
    }
    long countDerivated = 0;
    for (long i = 0; i < auxSize; i++)
        if (aux[i] >= INCLUDED)
            countDerivated++;
    return countDerivated;
}

__host__ __device__
long checkCaratheodorySetP3CSR(long *csrColIdxs, long nvertices,
        long *csrRowOffset, long sizeRowOffset,
        unsigned char *aux, unsigned char *auxc,
        long auxSize,
        long *currentCombination,
        long sizeComb, long idx, long *queue, long maxSizeQueue) {
    //clean aux vector            
    for (long i = 0; i < auxSize; i++) {
        aux[i] = 0;
        auxc[i] = 0;
    }

    long headQueue = 0;
    long tailQueue = -1;

    for (long i = 0; i < sizeComb; i++) {
        tailQueue = (tailQueue + 1) % maxSizeQueue;
        queue[tailQueue] = currentCombination[i];
        aux[currentCombination[i]] = INCLUDED;
        auxc[currentCombination[i]] = 1;
    }

    long countExec = 1;

    while (headQueue <= tailQueue) {
        if (verboseSerialDetailed) {
            printf("\nP3(k=%2d,c=%ld)-%ld: ", sizeComb, idx, countExec++);
            prlongQueue(queue, headQueue, tailQueue);
        }
        long verti = queue[headQueue];
        headQueue = (headQueue + 1) % maxSizeQueue;
        if (verboseSerialDetailed) {
            printf("\tv-rm: %d", verti);
        }

        if (verti < nvertices) {
            long end = csrColIdxs[verti + 1];
            for (long i = csrColIdxs[verti]; i < end; i++) {
                long vertn = csrRowOffset[i];
                if (vertn != verti && vertn < nvertices) {
                    unsigned char previousValue = aux[vertn];
                    aux[vertn] = aux[vertn] + NEIGHBOOR_COUNT_INCLUDED;
                    if (previousValue < INCLUDED) {
                        if (aux[vertn] >= INCLUDED) {
                            tailQueue = (tailQueue + 1) % maxSizeQueue;
                            queue[tailQueue] = vertn;
                        }
                        auxc[vertn] = auxc[vertn] + auxc[verti];
                    }
                }
            }
        }
    }

    long sizederivated = 0;

    for (long i = 0; i < auxSize; i++) {
        if (auxc[i] >= sizeComb) {
            sizederivated = calcDerivatedPartial(csrColIdxs, nvertices,
                    csrRowOffset, sizeRowOffset, aux, auxc, auxSize,
                    currentCombination, sizeComb, queue, maxSizeQueue);
            break;
        }
    }
    free(queue);
    return sizederivated;
}

long checkCaratheodorySetP3(UndirectedCSRGraph *graph,
        unsigned char *aux, unsigned char *auxc,
        long auxSize,
        long *currentCombination,
        long sizeComb, long idx) {
    long maxSizeQueue = MAX((auxSize / 2), MAX_DEFAULT_SIZE_QUEUE);
    long *queue = (long *) malloc(maxSizeQueue * sizeof (long));
    return checkCaratheodorySetP3CSR(graph->getCsrColIdxs(), graph->getVerticesCount(),
            graph->getCsrRowOffset(), graph->getSizeRowOffset(),
            aux, auxc, auxSize, currentCombination, sizeComb, idx, queue, maxSizeQueue);
}

void serialFindCaratheodoryNumberBinaryStrategy(UndirectedCSRGraph *graph) {
    graph->begin_serial_time = clock();

    long nvs = graph->getVerticesCount();
    long k;
    unsigned char *aux = new unsigned char [nvs];
    unsigned char *auxc = new unsigned char [nvs];
    long *currentCombination;

    long maxSizeSet = (nvs + 1) / 2;
    long sizeCurrentHcp3 = 0;

    long left = 0;
    long rigth = maxSizeSet;

    currentCombination = (long *) malloc(maxSizeSet * sizeof (long));
    long * lastCaratheodory = (long *) malloc(maxSizeSet * sizeof (long));
    long lastSize = -1;
    long lastSizeHcp3 = -1;

    while (left <= rigth) {
        k = (left + rigth) / 2;
        long maxCombination = maxCombinations(nvs, k);
        initialCombination(nvs, k, currentCombination);

        if (verboseSerial) {
            printf("\nserialFindCaratheodoryNumber: nvs=%d k=%d max=%d",
                    nvs, k, maxCombination);
        }

        bool found = false;
        for (long i = 0; i < maxCombination && !found; i++) {
            sizeCurrentHcp3 = checkCaratheodorySetP3(graph, aux, auxc, nvs, currentCombination, k, i);
            found = (sizeCurrentHcp3 > 0);
            if (!found) {
                nextCombination(nvs, k, currentCombination);
            }
        }
        if (found) {
            left = k + 1;
            lastSize = k;
            lastSizeHcp3 = sizeCurrentHcp3;
            for (long j = 0; j < k; j++) {
                lastCaratheodory[j] = currentCombination[j];
            }
            if (verboseSerial) {
                printf("\nCaratheodory set size=%d..Ok", k);
            }
        } else {
            rigth = k - 1;
            if (verboseSerial) {
                printf("\nCaratheodory set size=%d..Not", k);
            }
        }
    }
    if (lastSize > 0) {
        printf("\nResult");
        printCombination(lastCaratheodory, lastSize);
        printf("\n|S| = %d\n|∂H(S)| = %d", lastSize, lastSizeHcp3);
    }
    free(currentCombination);
    free(aux);
    graph->end_serial_time = clock();
}

void serialFindCaratheodoryNumber(UndirectedCSRGraph *graph) {
    graph->begin_serial_time = clock();

    long nvs = graph->getVerticesCount();
    long k;
    unsigned char *aux = new unsigned char [nvs];
    unsigned char *auxc = new unsigned char [nvs];
    long *currentCombination;

    long currentSize = (nvs + 1) / 2;
    long sizeCurrentHcp3 = 0;

    bool found = false;

    while (currentSize >= 2 && !found) {
        k = currentSize;
        long maxCombination = maxCombinations(nvs, k);
        currentCombination = (long *) malloc(k * sizeof (long));
        initialCombination(nvs, k, currentCombination);

        for (long i = 0; i < maxCombination && !found; i++) {
            sizeCurrentHcp3 = checkCaratheodorySetP3(graph, aux, auxc, nvs, currentCombination, k, i);
            found = (sizeCurrentHcp3 > 0);
            if (!found)
                nextCombination(nvs, k, currentCombination);
        }
        if (found) {
            printf("\nResult");
            printCombination(currentCombination, currentSize);
            printf("\n|S| = %d\n|∂H(S)| = %d", k, sizeCurrentHcp3);
        }
        currentSize--;
        free(currentCombination);
    }
    free(aux);
    graph->end_serial_time = clock();
}

__global__ void kernelFindCaratheodoryNumber(long *csrColIdxs, long nvertices,
        long *csrRowOffset, long sizeRowOffset, long maxCombination,
        long k, long offset, long *result, unsigned char *aux, unsigned char *auxc,
        unsigned long *cacheMaxCombination, long * queues, long maxSizeQueue, long *currentCombinations, long maxSizeCombination) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (verboseParallelDetailed)
        printf("\nThread-%d: szoffset=%d nvs=%d k=%d max=%d offset=%d", idx,
            sizeRowOffset, nvertices, k, maxCombination, offset);
    bool found = false;
    long auxoffset = idx*nvertices;
    long combinationoffset = idx*maxSizeCombination;
    long queueoffset = idx*maxSizeQueue;
    long sizederivated = 0;
    long limmit = (idx + 1) * offset;
    if (limmit > maxCombination) {
        limmit = maxCombination;
    }

    long i = idx * offset;


    if (verboseParallelDetailed)
        printf("\nThread-%d: k=%d(%d-%d)", idx, k, i, limmit);
    initialCombination(nvertices, k, &currentCombinations[combinationoffset], i, cacheMaxCombination);

    while (i < limmit && !found && !result[0]) {
        //        long headQueue = 0;
        //        long tailQueue = -1;

        //        for (long y = 0; y < nvertices; y++) {
        //            aux[auxoffset + y] = 0;
        //            auxc[auxoffset + y] = 0;
        //        }
        //
        //        for (long j = 0; j < k; j++) {
        //            tailQueue = (tailQueue + 1) % maxSizeQueue;
        //            queues[queueoffset + tailQueue] = currentCombinations[combinationoffset + j];
        //            aux[auxoffset + currentCombinations[combinationoffset + j]] = INCLUDED;
        //            auxc[auxoffset + currentCombinations[combinationoffset + j]] = 1;
        //        }
        //
        //        while (headQueue <= tailQueue && !result[0]) {
        //            long verti = queues[queueoffset + headQueue];
        //            headQueue = (headQueue + 1) % maxSizeQueue;
        //
        //            if (verti < nvertices) {
        //                long end = csrColIdxs[verti + 1];
        //                for (long x = csrColIdxs[verti]; x < end; x++) {
        //                    long vertn = csrRowOffset[x];
        //                    if (vertn != verti && vertn < nvertices) {
        //                        unsigned char previousValue = aux[auxoffset + vertn];
        //                        aux[auxoffset + vertn] = aux[auxoffset + vertn] + NEIGHBOOR_COUNT_INCLUDED;
        //                        if (previousValue < INCLUDED) {
        //                            if (aux[auxoffset + vertn] >= INCLUDED) {
        //                                tailQueue = (tailQueue + 1) % maxSizeQueue;
        //                                queues[queueoffset + tailQueue] = vertn;
        //                            }
        //                            auxc[auxoffset + vertn] = auxc[auxoffset + vertn] + auxc[auxoffset + verti];
        //                        }
        //                    }
        //                }
        //            }
        //        }
        //
        //        for (long z = 0; z < nvertices; z++) {
        //            if (auxc[auxoffset + z] >= k) {
        //                for (long t = 0; t < k; t++) {
        //                    long p = currentCombinations[combinationoffset + t];
        //                    long headQueue = 0;
        //                    long tailQueue = -1;
        //
        //                    for (long q = 0; q < nvertices; q++) {
        //                        auxc[auxoffset + q] = 0;
        //                    }
        //
        //                    for (long j = 0; j < k; j++) {
        //                        long v = currentCombinations[combinationoffset + j];
        //                        if (v != p) {
        //                            tailQueue = (tailQueue + 1) % maxSizeQueue;
        //                            queues[queueoffset + tailQueue] = v;
        //                            auxc[auxoffset + v] = INCLUDED;
        //                        }
        //                    }
        //                    while (headQueue <= tailQueue && !result[0]) {
        //                        long verti = queues[queueoffset + headQueue];
        //                        headQueue = (headQueue + 1) % maxSizeQueue;
        //                        aux[auxoffset + verti] = 0;
        //                        if (verti < nvertices) {
        //                            long end = csrColIdxs[verti + 1];
        //                            for (long w = csrColIdxs[verti]; w < end; w++) {
        //                                long vertn = csrRowOffset[w];
        //                                if (vertn != verti) {
        //                                    int previousValue = auxc[auxoffset + vertn];
        //                                    auxc[auxoffset + vertn] = auxc[auxoffset + vertn] + NEIGHBOOR_COUNT_INCLUDED;
        //                                    if (previousValue < INCLUDED && auxc[auxoffset + vertn] >= INCLUDED) {
        //                                        tailQueue = (tailQueue + 1) % maxSizeQueue;
        //                                        queues[queueoffset + tailQueue] = vertn;
        //                                    }
        //                                }
        //                            }
        //                        }
        //                    }
        //                }
        //                sizederivated = 0;
        //                for (long j = 0; j < nvertices; j++)
        //                    if (aux[auxoffset + j] >= INCLUDED)
        //                        sizederivated++;
        //                break;
        //            }
        //        }
        sizederivated = checkCaratheodorySetP3CSR(csrColIdxs, nvertices, csrRowOffset, sizeRowOffset,
                &aux[auxoffset], &auxc[auxoffset], nvertices, &currentCombinations[combinationoffset],
                k, i, &queues[queueoffset], maxSizeQueue);

        found = (sizederivated > 0);
        if (!found) {
            nextCombination(nvertices, k, &currentCombinations[combinationoffset]);
            i++;
        }
        //        else {
        //            result[0] = sizederivated;
        //            result[1] = i;
        //            if (verboseParallelDetailed)
        //                printf("\nParallel find");
        //        }
    }
    if (found) {
        result[0] = sizederivated;
        result[1] = i;
        if (verboseParallelDetailed)
            printf("\nParallel find");
    }
}

void parallelFindCaratheodoryNumberBinaryStrategy(UndirectedCSRGraph *graph) {
    graph->begin_parallel_time = clock();
    clock_t parcial = clock();

    long result[2];
    result[0] = result[1] = 0;
    long* csrColIdxs = graph->getCsrColIdxs();
    long verticesCount = graph->getVerticesCount();
    long* csrRowOffset = graph->getCsrRowOffset();
    long sizeRowOffset = graph->getSizeRowOffset();

    long* csrColIdxsGpu;
    long* csrRowOffsetGpu;
    long *resultGpu;
    unsigned char *auxGpu;
    unsigned char *auxcGpu;
    unsigned long *cacheMaxCombination;

    long numBytesClsIdx = sizeof (long)*(verticesCount + 1);
    cudaMalloc((void**) &csrColIdxsGpu, numBytesClsIdx);

    long numBytesRowOff = sizeof (long)*sizeRowOffset;
    cudaMalloc((void**) &csrRowOffsetGpu, numBytesRowOff);

    long *currentCombinations;
    long *queues;
    long maxSizeQueue = MAX(verticesCount / 2, MAX_DEFAULT_SIZE_QUEUE);
    long maxCombinationSize = (verticesCount + 1 / 2);
    long memoryQueue = sizeof (long) * DEFAULT_THREAD_PER_BLOCK * maxSizeQueue;
    cudaMalloc((void**) &queues, memoryQueue);
    long memoryCombination = sizeof (long) * DEFAULT_THREAD_PER_BLOCK * maxCombinationSize;
    cudaMalloc((void**) &currentCombinations, memoryCombination);
    long memoryAux = sizeof (unsigned char) * DEFAULT_THREAD_PER_BLOCK * verticesCount;
    cudaMalloc((void**) &auxGpu, memoryAux);
    long memoryAuxc = sizeof (unsigned char) * DEFAULT_THREAD_PER_BLOCK * verticesCount;
    cudaMalloc((void**) &auxcGpu, memoryAuxc);
    long memoryCache = sizeof (unsigned long) * verticesCount;
    cudaMalloc((void**) &cacheMaxCombination, memoryCache);
    cudaMemset(cacheMaxCombination, 0, sizeof (unsigned long) * verticesCount);

    //    memoryQueue + memoryCombination + memoryAux + memoryAuxc + memoryCache + numBytesClsIdx + numBytesRowOff

    long numBytesResult = sizeof (long)*2;
    cudaMalloc((void**) &resultGpu, numBytesResult);

    if (resultGpu == NULL || csrRowOffsetGpu == NULL || csrColIdxsGpu == NULL) {
        perror("Failed allocate memory in GPU");
    }

    cudaError_t r = cudaMemcpy(csrColIdxsGpu, csrColIdxs, numBytesClsIdx, cudaMemcpyHostToDevice);
    if (r != cudaSuccess) {
        perror("Failed to copy memory");
    }
    r = cudaMemcpy(csrRowOffsetGpu, csrRowOffset, numBytesRowOff, cudaMemcpyHostToDevice);
    if (r != cudaSuccess) {
        perror("Failed to copy memory");
    }
    r = cudaMemcpy(resultGpu, result, numBytesResult, cudaMemcpyHostToDevice);
    if (r != cudaSuccess) {
        perror("Failed to copy memory");
    }

    long maxSizeSet = (verticesCount + 1) / 2;

    long k = 0;
    long left = 0;
    long rigth = maxSizeSet;

    long lastSize = -1;
    long lastIdxCaratheodorySet = -1;
    long lastSizeHcp3 = -1;
    clock_t start;
    while (left <= rigth) {
        k = (left + rigth) / 2;
        long maxCombination = maxCombinations(verticesCount, k);
        long threadsPerBlock = MIN(ceil(maxCombination / 3.0), DEFAULT_THREAD_PER_BLOCK);
        long offset = ceil(maxCombination / (double) threadsPerBlock);

        if (verboseParallel) {
            start = clock();
            printf("\nkernelFindCaratheodoryNumber: szoffset=%d nvs=%d k=%d max=%d tdsPerBlock=%d offset=%d",
                    sizeRowOffset, verticesCount, k, maxCombination, threadsPerBlock, offset);
        }

        //        if (k > 1) {
        //        kernelFindCaratheodoryNumber << <1, threadsPerBlock, memoryQueue + memoryCombination + memoryAux + memoryAuxc + memoryCache + numBytesClsIdx + numBytesRowOff + numBytesResult>>>(csrColIdxsGpu, verticesCount, csrRowOffsetGpu,
        //                sizeRowOffset, maxCombination, k, offset, resultGpu, auxGpu, auxcGpu, cacheMaxCombination,
        //                queues, maxSizeQueue, currentCombinations, maxCombinationSize);
        kernelFindCaratheodoryNumber << <1, threadsPerBlock>>>(csrColIdxsGpu, verticesCount, csrRowOffsetGpu,
                sizeRowOffset, maxCombination, k, offset, resultGpu, auxGpu, auxcGpu, cacheMaxCombination,
                queues, maxSizeQueue, currentCombinations, maxCombinationSize);
        cudaMemcpy(result, resultGpu, numBytesResult, cudaMemcpyDeviceToHost);
        //        } else {
        //            result[0] = 1;
        //            result[1] = 0;
        //        }

        if (verboseParallel) {
            printf(" t=%d", ((clock() - start) / (CLOCKS_PER_SEC / 1000)));
        }

        if (result[0] > 0) {
            lastSize = k;
            lastSizeHcp3 = result[0];
            lastIdxCaratheodorySet = result[1];
            left = k + 1;
            result[0] = result[1] = 0;
            cudaMemcpy(resultGpu, result, numBytesResult, cudaMemcpyHostToDevice);
            if (verboseParallel) {
                printf("\nCaratheodory set size=%d..Ok idx=%d", k, lastIdxCaratheodorySet);
            }
        } else {
            rigth = k - 1;
            if (verboseParallel) {
                printf("\nCaratheodory set size=%d..Not", k);
            }
        }
    }
    if (lastSize > 0) {
        printf("\nResult Parallel Binary");
        long *currentCombination = (long *) malloc(lastSize * sizeof (long));
        initialCombination(verticesCount, lastSize, currentCombination, lastIdxCaratheodorySet);
        printCombination(currentCombination, lastSize);
        printf("\nS=%d-Comb(%d,%d) \n|S| = %d",
                lastIdxCaratheodorySet, verticesCount, lastSize, lastSize);
    } else {
        printf("\nCaratheodory set not found!");
    }
    cudaFree(resultGpu);
    cudaFree(csrRowOffsetGpu);
    cudaFree(csrColIdxsGpu);
    cudaFree(auxGpu);
    cudaFree(auxcGpu);
    cudaFree(cacheMaxCombination);
    cudaFree(queues);
    cudaFree(currentCombinations);
    graph->end_parallel_time = clock();
}

void parallelFindCaratheodoryNumber(UndirectedCSRGraph *graph) {
    graph->begin_parallel_time = clock();
    clock_t parcial = clock();

    long nvs = graph->getVerticesCount();
    long currentSize = 0;
    long result[2];
    result[0] = result[1] = 0;
    long* csrColIdxs = graph->getCsrColIdxs();
    long verticesCount = graph->getVerticesCount();
    long* csrRowOffset = graph->getCsrRowOffset();
    long sizeRowOffset = graph->getSizeRowOffset();

    long* csrColIdxsGpu;
    long* csrRowOffsetGpu;
    long *resultGpu;
    unsigned char *auxGpu;
    unsigned char *auxcGpu;
    unsigned long *cacheMaxCombination;

    long numBytesClsIdx = sizeof (long)*(verticesCount + 1);
    cudaMalloc((void**) &csrColIdxsGpu, numBytesClsIdx);

    long numBytesRowOff = sizeof (long)*sizeRowOffset;
    cudaMalloc((void**) &csrRowOffsetGpu, numBytesRowOff);

    long *currentCombinations;
    long *queues;
    long maxSizeQueue = MAX(verticesCount / 2, MAX_DEFAULT_SIZE_QUEUE);
    long maxCombinationSize = (verticesCount + 1 / 2);
    cudaMalloc((void**) &queues, sizeof (long) * DEFAULT_THREAD_PER_BLOCK * maxSizeQueue);
    cudaMalloc((void**) &currentCombinations, sizeof (long) * DEFAULT_THREAD_PER_BLOCK * maxCombinationSize);
    cudaMalloc((void**) &auxGpu, sizeof (unsigned char)*DEFAULT_THREAD_PER_BLOCK * nvs);
    cudaMalloc((void**) &auxcGpu, sizeof (unsigned char)*DEFAULT_THREAD_PER_BLOCK * nvs);
    cudaMalloc((void**) &cacheMaxCombination, sizeof (unsigned long)*nvs);
    cudaMemset(cacheMaxCombination, 0, sizeof (unsigned long)*nvs);

    long numBytesResult = sizeof (long)*2;
    cudaMalloc((void**) &resultGpu, numBytesResult);

    if (resultGpu == NULL || csrRowOffsetGpu == NULL || csrColIdxsGpu == NULL) {
        perror("Failed allocate memory in GPU");
    }

    cudaError_t r = cudaMemcpy(csrColIdxsGpu, csrColIdxs, numBytesClsIdx, cudaMemcpyHostToDevice);
    if (r != cudaSuccess) {
        perror("Failed to copy memory");
    }
    r = cudaMemcpy(csrRowOffsetGpu, csrRowOffset, numBytesRowOff, cudaMemcpyHostToDevice);
    if (r != cudaSuccess) {
        perror("Failed to copy memory");
    }
    r = cudaMemcpy(resultGpu, result, numBytesResult, cudaMemcpyHostToDevice);
    if (r != cudaSuccess) {
        perror("Failed to copy memory");
    }

    bool found = false;
    currentSize = (nvs + 1) / 2;

    while (currentSize >= 2 && !found) {
        long maxCombination = maxCombinations(nvs, currentSize);
        long threadsPerBlock = MIN(ceil(maxCombination / 3.0), DEFAULT_THREAD_PER_BLOCK);
        long offset = ceil(maxCombination / (double) threadsPerBlock);

        if (verboseParallel)
            printf("\nkernelFindCaratheodoryNumber: szoffset=%d nvs=%d k=%d max=%d offset=%d\n",
                sizeRowOffset, verticesCount, currentSize, maxCombination, offset);
        kernelFindCaratheodoryNumber << <1, threadsPerBlock>>> (csrColIdxsGpu, verticesCount,
                csrRowOffsetGpu, sizeRowOffset, maxCombination, currentSize, offset, resultGpu,
                auxGpu, auxcGpu, cacheMaxCombination,
                queues, maxSizeQueue, currentCombinations, maxCombinationSize);

        cudaMemcpy(result, resultGpu, numBytesResult, cudaMemcpyDeviceToHost);
        found = (result[0] > 0);
        currentSize--;
    }
    if (found) {
        printf("Result Parallel\n");
        long *currentCombination = (long *) malloc((currentSize + 1) * sizeof (long));
        initialCombination(verticesCount, (currentSize + 1), currentCombination, result[1]);
        printCombination(currentCombination, currentSize + 1);
        printf("\nS=%d-Comb(%d,%d) \n|S| = %d \n|∂H(S)| = %d\n",
                result[1], nvs, currentSize + 1, currentSize + 1, result[0]);
    } else {
        printf("Caratheodory set not found!\n");
    }
    cudaFree(resultGpu);
    cudaFree(csrRowOffsetGpu);
    cudaFree(csrColIdxsGpu);
    cudaFree(auxGpu);
    cudaFree(cacheMaxCombination);
    graph->end_parallel_time = clock();
}
