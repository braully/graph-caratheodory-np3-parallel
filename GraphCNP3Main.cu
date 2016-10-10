#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <string>
#include <sstream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "UndirectedSparseGraph.h"
#include "GraphCaratheodoryNumber.h"

#define CHARACTER_INIT_COMMENT '#'

//References and Examples:
//http://www.boost.org/doc/libs/1_39_0/libs/graph/example/undirected.cpp
//Example: http://devblogs.nvidia.com/parallelforall/wp-content/uploads/2014/07/CSR.png
//Example: https://devblogs.nvidia.com/parallelforall/accelerating-graph-betweenness-centrality-cuda/


bool verboseGraph = false;
//bool verboseCsr = false;

void printHelp() {
    printf("\n\tp: Parallel execution");
    printf("\n\ts: Serial execution");
    printf("\n\th: Prlong this help message");
    printf("\n\tinput: Graph file in format CSR, see graph-test.txt");
}

void processFile(std::string strFile, bool serial, bool parallel, bool verbose) {
    std::string line, strCArray, strRArray;
    std::ifstream infile(strFile.c_str());

    if (infile) {
        while (getline(infile, line)) {
            if (line.at(0) != CHARACTER_INIT_COMMENT) {
                if (strCArray.empty()) {
                    strCArray = line;
                } else if (strRArray.empty()) {
                    strRArray = line;
                }
            }
        }
    } else {
        printf("file '%s' not found!", strFile.c_str());
        return;
    }

    infile.close();

    if (strCArray.empty() || strRArray.empty()) {
        perror("Invalid file format");
        return;
    }

    std::stringstream stream(strCArray);
    std::vector<int> values;
    long n;
    while (stream >> n) {
        values.push_back(n);
    }
    strCArray.clear();

    long numVertices = values.size() - 1;
    long *colIdx = new long[numVertices + 1];
    std::copy(values.begin(), values.end(), colIdx);
    values.clear();
    stream.str("");

    std::stringstream stream2(strRArray);
    while (stream2 >> n) {
        values.push_back(n);
    }
    stream2.str("");
    strRArray.clear();

    long sizeRowOffset = values.size();
    long *rowOffset = new long[sizeRowOffset];
    std::copy(values.begin(), values.end(), rowOffset);
    values.clear();

    UndirectedCSRGraph csr(colIdx, numVertices, rowOffset, sizeRowOffset);
    if (verbose) {
        printf("Graph detail: ");
        csr.printGraph();
        printf("\n");
    }

    printf("Process file: %s\n", strFile.c_str());

    if (serial) {
        serialFindCaratheodoryNumber(&csr);
        printf("Total time serial: %ldms\n",
                csr.getTotalTimeSerial());
    }
    if (parallel) {
        parallelFindCaratheodoryNumber(&csr);
        printf("Total time parallel: %ldms\n",
                csr.getTotalTimeParallel());
    }

    if (serial && parallel) {
        double speedup = (double) csr.getTotalTimeSerial() / csr.getTotalTimeParallel();
        printf("Speedup: %fx\n", speedup);

    }
}

int main(int argc, char** argv) {
    long opt = 0;
    //    char* strFile = "graph-test";
    char* strFile = "graph-test/graph-csr-8566306576678665541.txt";
    bool serial = false;
    bool parallel = false;
    bool verbose = false;

    if ((argc <= 1) || (argv[argc - 1] == NULL) || (argv[argc - 1][0] == '-')) {
        serial = true;
        parallel = true;
        //        printf("\nusage: graph-hull-number [-ps] [input]");
        //        printHelp();
        //        return 1;
    } else {
        strFile = argv[argc - 1];
    }

    while ((opt = getopt(argc, argv, "psv")) != -1) {
        switch (opt) {
            case 'p':
                parallel = true;
                break;
            case 's':
                serial = true;
                break;
            case 'v':
                verbose = true;
                break;
            case '?':
                printf("Unknow option: %c", char(opt));
                break;
        }
    }
    DIR *dpdf;
    struct dirent *epdf;
    struct stat filestat;

    dpdf = opendir(strFile);
    if (dpdf != NULL) {
        while (epdf = readdir(dpdf)) {
            std::string filepath = std::string(strFile) + "/" + epdf->d_name;
            if (epdf->d_name == "." || epdf->d_name == "..")
                continue;
            if (stat(filepath.c_str(), &filestat)) continue;
            if (S_ISDIR(filestat.st_mode)) continue;
            processFile(filepath, serial, parallel, verbose);
        }
        closedir(dpdf);
    } else {
        processFile(strFile, serial, parallel, verbose);
    }
    return 0;
}
