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

using namespace std;

// predictions is the array where you have to return the class predicted (integer) for the test dataset instances
int* predictions;

// Arguments for each thread with divided data
struct arguments {
    ArffData* train;
    ArffData* test;
    int k;
    int start; 
    int end;
};

float distance(ArffInstance* a, ArffInstance* b) {
    float sum = 0;
    
    for (int i = 0; i < a->size()-1; i++) {
        float diff = (a->get(i)->operator float() - b->get(i)->operator float());
        sum += diff*diff;
    }
    
    return sum;
}

void* KNN(void *params) {
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.

    struct arguments *args = (struct arguments*) params;

    ArffData *train = args->train;
    ArffData *test = args->test;
    int k = args->k;
    int start = args->start;
    int end = args->end;

    // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float* candidates = (float*) calloc(k*2, sizeof(float));
    for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }

    int num_classes = train->num_classes();

    // Stores bincounts of each class over the final set of candidate NN
    int* classCounts = (int*) calloc(num_classes, sizeof(int));

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

        predictions[queryIndex] = max_index;

        for(int i = 0; i < 2*k; i++) { 
            candidates[i] = FLT_MAX; 
        }

        memset(classCounts, 0, num_classes * sizeof(int));
    }

    pthread_exit(0);
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

    if(argc != 5)
    {
        cout << "Usage: ./multi-thread datasets/train.arff datasets/test.arff k numThreads" << endl;
        exit(0);
    }

    int k = strtol(argv[3], NULL, 10);

    int numThreads = strtol(argv[4], NULL, 10);
    pthread_t *threads;
  
    threads = (pthread_t*) malloc(numThreads * sizeof(pthread_t));

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();

    // Divide data per thread with leftover in last thread
    int datasetSize = test->num_instances();
    int dataPerThread = datasetSize / numThreads;
    int leftoverData = datasetSize % numThreads;
    int dataForLastThread = dataPerThread + leftoverData;
    
    predictions = (int*) malloc(test->num_instances() * sizeof(int));
    int dataStart = 0, dataEnd = 0;

    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    struct arguments *params;

    // Make threads with specific args for KNN function
    for(int i = 0; i < numThreads; i++) {
        params = (struct arguments*) malloc(sizeof(struct arguments));

        if(i < numThreads - 1) {
            dataEnd = dataStart + dataPerThread;
        } else {
            dataEnd = dataStart + dataForLastThread;
        }

        params->train = train;
        params->test = test;
        params->k = k;
        params->start = dataStart;
        params->end = dataEnd;

        pthread_create(&threads[i], NULL, KNN, (void*) params);

        dataStart = dataEnd;
    }
    
    for(int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL); 
    }
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The %i-NN classifier for %lu test instances on %lu train instances required %llu ms CPU time. Accuracy was %.4f\n", k, test->num_instances(), train->num_instances(), (long long unsigned int) diff, accuracy);

    free(threads);
    free(predictions);
    free(params);
}
