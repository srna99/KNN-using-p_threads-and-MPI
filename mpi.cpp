#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <tuple>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

float distance(ArffInstance* a, ArffInstance* b) {
    float sum = 0;
    
    for (int i = 0; i < a->size()-1; i++) {
        float diff = (a->get(i)->operator float() - b->get(i)->operator float());
        sum += diff*diff;
    } 
    
    return sum;
}

int* KNN(ArffData* train, ArffData* test, int k, int start, int end) {
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.

    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int* predictions = (int*) malloc((end - start) * sizeof(int));

    // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float* candidates = (float*) calloc(k*2, sizeof(float));
    for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }

    int num_classes = train->num_classes();

    // Stores bincounts of each class over the final set of candidate NN
    int* classCounts = (int*) calloc(num_classes, sizeof(int));
    int predictionIndex = 0;

    for(int queryIndex = start; queryIndex < end; queryIndex++) {
        for(int keyIndex = 0; keyIndex < train->num_instances(); keyIndex++) {
            float dist = distance(test->get_instance(queryIndex), train->get_instance(keyIndex));

            // Add to our candidates
            for(int c = 0; c < k; c++) {
                if(dist < candidates[2*c]) {
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for(int x = k-2; x >= c; x--) {
                        candidates[2*x+2] = candidates[2*x];
                        candidates[2*x+3] = candidates[2*x+1];
                    }
                    
                    // Set key vector as potential k NN
                    candidates[2*c] = dist;
                    candidates[2*c+1] = train->get_instance(keyIndex)->get(train->num_attributes() - 1)->operator float(); // class value

                    break;
                }
            }
        }

        // Bincount the candidate labels and pick the most common
        for(int i = 0; i < k; i++) {
            classCounts[(int)candidates[2*i+1]] += 1;
        }

        int max = -1;
        int max_index = 0;
        for(int i = 0; i < num_classes;i++) {
            if(classCounts[i] > max) {
                max = classCounts[i];
                max_index = i;
            }
        }

        predictions[predictionIndex] = max_index;
        predictionIndex += 1;

        for(int i = 0; i < 2*k; i++) { 
            candidates[i] = FLT_MAX; 
        }

        memset(classCounts, 0, num_classes * sizeof(int));
    }

    return predictions;
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
          
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[]){

    if(argc != 4)
    {
        cout << "Usage: mpiexec -np numProcesses ./mpi datasets/train.arff datasets/test.arff k" << endl;
        exit(0);
    }

    int k = strtol(argv[3], NULL, 10);
    int worldRank, worldSize, tag = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Status stat;

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();

    // Divide data per process with leftover in last process
    int datasetSize = test->num_instances();
    int dataPerProcess = datasetSize / worldSize;
    int leftoverData = datasetSize % worldSize;
    int dataForLastProcess = dataPerProcess + leftoverData;
    
    int* predictions = (worldRank == 0) ? (int*) malloc(datasetSize * sizeof(int)) : NULL;
    int* subPredictions = NULL;
    int dataStart = 0, dataEnd = 0;

    int scatterSendBuffer[worldSize][2];
    int scatterRecvBuffer[2];

    struct timespec start, end;

    if(worldRank == 0) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);

        for(int i = 0; i < worldSize; i++) {
            if(i < worldSize - 1) {
                dataEnd = dataStart + dataPerProcess;
            } else {
                dataEnd = dataStart + dataForLastProcess;
            }

            scatterSendBuffer[i][0] = dataStart;
            scatterSendBuffer[i][1] = dataEnd;

            dataStart = dataEnd;
        }
    } 

    MPI_Scatter(scatterSendBuffer, 2, MPI_INT, scatterRecvBuffer, 2, MPI_INT, 0, MPI_COMM_WORLD);

    subPredictions = KNN(train, test, k, scatterRecvBuffer[0], scatterRecvBuffer[1]);

    int gatherSendCount = (worldRank != worldSize - 1) ? dataPerProcess : dataForLastProcess;
    int* gatherRecvCounts = (int*) malloc(worldSize * sizeof(int));
    int* gatherRecvDisplacements = (int*) malloc(worldSize * sizeof(int));

    for(int i = 0; i < worldSize; i++) {
        gatherRecvCounts[i] = (i != worldSize - 1) ? dataPerProcess : dataForLastProcess;
        gatherRecvDisplacements[i] = i * dataPerProcess;
    }

    MPI_Gatherv(subPredictions, gatherSendCount, MPI_INT, predictions, gatherRecvCounts, gatherRecvDisplacements, MPI_INT, 0, MPI_COMM_WORLD);

    if(worldRank == 0) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        // Compute the confusion matrix
        int* confusionMatrix = computeConfusionMatrix(predictions, test);
        // Calculate the accuracy
        float accuracy = computeAccuracy(confusionMatrix, test);

        uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

        printf("The %i-NN classifier for %lu test instances on %lu train instances required %llu ms CPU time. Accuracy was %.4f\n", k, test->num_instances(), train->num_instances(), (long long unsigned int) diff, accuracy);
    }

    MPI_Finalize();

    if(worldRank == 0) { free(predictions); }
    free(gatherRecvCounts);
    free(gatherRecvDisplacements);
}
