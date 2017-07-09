// ./copyright
//
// INTEL CONFIDENTIAL 
//
// Copyright 2011 Intel Corporation All Rights Reserved.  
//
// The source code contained or described herein and all documents related to the 
// source code ("Material") are owned by Intel Corporation or its suppliers
// or licensors. Title to the Material remains with Intel Corporation or its suppliers 
// and licensors. The Material contains trade secrets and proprietary and confidential 
// information of Intel or its suppliers and licensors. The Material is protected by 
// worldwide copyright and trade secret laws and treaty provisions. No part of the 
// Material may be used, copied, reproduced, modified, published, uploaded, posted,
// transmitted, distributed, or disclosed in any way without Intel.s prior express 
// written permission.
//
// No license under any patent, copyright, trade secret or other intellectual property 
// right is granted to or conferred upon you by disclosure or delivery of the Materials, 
// either expressly, by implication, inducement, estoppel or otherwise. Any license under 
// such intellectual property rights must be express and approved by Intel in writing.
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "black-scholes.h"
#include <iostream>
#include <mpi.h>
using namespace std;

inline float RandFloat(float low, float high) {
	float t = (float) rand() / (float) RAND_MAX;
	return (1.0f - t) * low + t * high;
}

int main(int argc, char* argv[]) { 
	FPARRAY CallResultParallel, CallConfidence, StockPrice, OptionStrike,
			OptionYears,res,subres;
	
	MPI_Init (&argc, &argv);	

	int i, mem_size;

	int OPT_N;

	int myrank, p;
	int tag = 0;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);


	
	
	if (myrank == 0){

	scanf("%d",&OPT_N);
	}


	MPI_Bcast (&OPT_N, 1, MPI_INT, 0, MPI_COMM_WORLD);

	printf("meu id %d - valor de OPT_N: %d \n", myrank, OPT_N);

	mem_size = sizeof(float) * OPT_N;

	res.SPData = (float *) malloc(mem_size);
	int subTam =(OPT_N/p);
	subres.SPData = (float *) malloc(mem_size/p);

	CallResultParallel.SPData = (float *) malloc(mem_size);
	CallConfidence.SPData = (float *) malloc(mem_size);
	StockPrice.SPData = (float *) malloc(mem_size);
	OptionStrike.SPData = (float *) malloc(mem_size);
	OptionYears.SPData = (float *) malloc(mem_size);

	if (myrank==0){

		for (i = 0; i < OPT_N; i++) {
			CallResultParallel.SPData[i] = 0.0;
			CallConfidence.SPData[i] = -1.0;
			StockPrice.SPData[i] = RandFloat(5.0f, 50.0f);
			OptionStrike.SPData[i] = RandFloat(10.0f, 25.0f);
			OptionYears.SPData[i] = RandFloat(1.0f, 5.0f);
		}
	}

	MPI_Bcast (CallResultParallel.SPData, OPT_N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast (CallConfidence.SPData, OPT_N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast (StockPrice.SPData, OPT_N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast (OptionStrike.SPData, OPT_N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast (OptionYears.SPData, OPT_N, MPI_FLOAT, 0, MPI_COMM_WORLD);


	MonteCarlo(CallResultParallel.SPData, CallConfidence.SPData,
			StockPrice.SPData, OptionStrike.SPData, OptionYears.SPData, OPT_N, myrank, p);
	int aux=0;
	for (int opt = myrank*(OPT_N/p); opt < (myrank+1)*(OPT_N/p); opt++) {
		subres.SPData[aux++]=CallResultParallel.SPData[opt];
	}
	MPI_Gather(subres.SPData,(OPT_N/p), MPI_FLOAT,CallResultParallel.SPData,(OPT_N/p), MPI_FLOAT, 0, MPI_COMM_WORLD);
	if(myrank==0)
	{
		for (i = 0; i < OPT_N; i++){
			printf( "%d - [%f] \n", myrank, CallResultParallel.SPData[i]);	
		}
	}
	
	free(CallResultParallel.SPData);
	free(CallConfidence.SPData);
	free(StockPrice.SPData);
	free(OptionStrike.SPData);
	free(OptionYears.SPData);

	MPI_Finalize();
	return 0;
}
