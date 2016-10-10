/* 
 * File:   UndirectedSparseGraph.h
 * Author: braully
 *
 * Created on 10 de Junho de 2016, 09:52
 */
#define PROCESSED 3
#define INCLUDED 2
#define NEIGHBOOR_COUNT_INCLUDED 1
//
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <time.h>
#include <iostream>

class UndirectedSparseGraph {
public:
    clock_t begin_serial_time, end_serial_time;
    clock_t begin_parallel_time, end_parallel_time;

    UndirectedSparseGraph(const long size) {
        begin_serial_time = end_serial_time = 0;
        begin_parallel_time = end_parallel_time = 0;
        vertices = (long *) malloc(size * sizeof (long));
        adj_list = new std::list < long > [size];
        nVertices = size;
        for (long i = 0; i < size; i++) {
            vertices[i] = i;
        }
    }

    void addEdge(long v1, long v2) {
        if (v1 >= nVertices || v2 >= nVertices) {
            perror("Invalid vertices!");
        }
        adj_list[vertices[v1]].push_back(vertices[v2]);
        adj_list[vertices[v2]].push_back(vertices[v1]);
        //        vertices[v1].
    }

    void prlongGraph() {
        printf("\nVertices: {");
        for (long i = 0; i < nVertices; i++) {
            printf("v%d(d%ld)", vertices[i], adj_list[i].size());
            if (i < nVertices - 1) {
                printf(", ");
            }
        }
        printf("}\n");
    }

    long degree(long vert) {
        if (vert >= nVertices) {
            perror("Invalid vertices!");
        }
        return adj_list[vert].size();
    }

    long getVerticesCount() {
        return nVertices;
    }

    std::list<long> getAdjacency(long vert) {
        if (vert > nVertices) {
            perror("Invalid vertice");
        }
        return adj_list[vert];
    }

    void clear() {

    }
private:
    long *vertices;
    long nVertices;
    std::list < long >* adj_list;
};

class UndirectedCSRGraph {
public:
    clock_t begin_serial_time, end_serial_time;
    clock_t begin_parallel_time, end_parallel_time;

    clock_t getTotalTimeSerial() {
        return diffclockms(end_serial_time, begin_serial_time);
    }

    clock_t getTotalTimeParallel() {
        return diffclockms(end_parallel_time, begin_parallel_time);
    }

    double diffclockms(clock_t c1, clock_t c2) {
        double diff = c1 - c2;
        diff = diff / (CLOCKS_PER_SEC / 1000);
        return diff;
    }

    UndirectedCSRGraph(long *csrClIdxs, long nVerts,
            long *csrRowOffst, long sizeRowOffst) {
        begin_serial_time = end_serial_time = 0;
        begin_parallel_time = end_parallel_time = 0;
        nVertices = nVerts;
        sizeRowOffset = sizeRowOffst;
        csrColIdxs = csrClIdxs;
        csrRowOffset = csrRowOffst;
    }

    UndirectedCSRGraph(UndirectedSparseGraph* graph) {
        begin_serial_time = end_serial_time = 0;
        begin_parallel_time = end_parallel_time = 0;
        nVertices = graph->getVerticesCount();
        sizeRowOffset = 0;
        csrColIdxs = new long[nVertices + 1];
        for (long i = 0; i < nVertices; i++) {
            csrColIdxs[i] = 0;
            sizeRowOffset = sizeRowOffset + graph->degree(i);
        }

        csrRowOffset = new long[sizeRowOffset];
        long idx = 0;
        for (long i = 0; i < nVertices; i++) {
            csrColIdxs[i] = idx;
            std::list<long>::iterator it;
            std::list<long> tmp = graph->getAdjacency(i);
            for (std::list<long>::iterator it = tmp.begin(); it != tmp.end(); ++it) {
                //                if (verboseCsr) {
                //                    printf("csrRow[%d] = %d\n", idx, *it);
                //                }
                csrRowOffset[idx++] = *it;
            }
        }
        csrColIdxs[nVertices] = idx;
    }

    long *getCsrColIdxs() {
        return csrColIdxs;
    }

    long *getCsrRowOffset() {
        return csrRowOffset;
    }

    long getOffsetBeginNeighbors(long vert) {
        if (vert >= nVertices) {
            perror("Invalid vertices!");
        }
        return csrColIdxs[vert];
    }

    long getIdxAdjList(long idx) {
        return csrRowOffset[idx];
    }

    long getOffsetEndNeighbors(long vert) {
        if (vert >= nVertices) {
            perror("Invalid vertices!");
        }
        return csrColIdxs[vert + 1];
    }

    long degree(long vert) {
        if (vert >= nVertices) {
            perror("Invalid vertices!");
        }
        return csrColIdxs[vert + 1] - csrColIdxs[vert];
    }

    long getVerticesCount() {
        return nVertices;
    }

    long getSizeRowOffset() {
        return sizeRowOffset;
    }

    void printGraph() {
        printf("\nN. vertices: %d", nVertices);
        printf("\nSize RowOffset: %d", sizeRowOffset);
        printf("\nR: {");
        for (long i = 0; i < nVertices + 1; i++) {
            printf("%d", csrColIdxs[i]);
            if (i < nVertices) {
                printf(", ");
            }
        }
        printf("} \nC: {");
        for (long i = 0; i < sizeRowOffset; i++) {
            printf("%d", csrRowOffset[i]);
            if (i < sizeRowOffset - 1) {
                printf(", ");
            }
        }
        printf("}");
    }
    //    virtual ~UndirectedCSRGraph();
private:
    long nVertices;
    long sizeRowOffset;
    long *csrColIdxs;
    long *csrRowOffset;
};

