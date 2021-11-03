#include "mpi.h"
#include <malloc.h>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>
int myid, numprocs, Root = 0, * sendcounts, * displs;
double* AB, * A, * X;
int size;
int MATR_SIZE;
int SIZE;
int ind(int i, int j, int SIZE) {
    return (i * (SIZE + 1) + j);
}
void jacoby(double* X_old, int size, int MATR_SIZE, int first) {
    int i, j;
    double Sum;
    for (i = 0; i < size; i++) {
        Sum = 0;
        for (j = 0; j < i + first; ++j) {
            Sum += A[ind(i, j, MATR_SIZE)] * X_old[j];
        }
        for (j = i + 1 + first; j < MATR_SIZE; ++j) {
            Sum += A[ind(i, j, MATR_SIZE)] * X_old[j];
        }
        X[i + first] = (A[ind(i, MATR_SIZE, MATR_SIZE)] - Sum) / A[ind(i, i + first, MATR_SIZE)];
    }
}
void solve(int MATR_SIZE, int size, double Error) {
    double* X_old;
    int Iter = 0, i, Result, first;
    double diff_norm = 0, diff_value;
    MPI_Scan(&size, &first, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    first -= size;
    MPI_Allgather(&size, 1, MPI_INT, sendcounts, 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (i = 1; i < numprocs; ++i)
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    X_old = (double*)malloc(sizeof(double) * MATR_SIZE);
    do {
        ++Iter;
        memcpy(X_old, X, sizeof(double) * MATR_SIZE);
        jacoby(X_old, size, MATR_SIZE, first);
        MPI_Allgatherv(&X[first], size, MPI_DOUBLE, X, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        if (myid == Root) {
            diff_norm = 0;
            for (i = 0; i < MATR_SIZE; ++i) {
                diff_value = fabs(X[i] - X_old[i]);
                if (diff_norm < diff_value) diff_norm = diff_value;
            }
            Result = Error < diff_norm;
        }
        MPI_Bcast(&Result, 1, MPI_INT, Root, MPI_COMM_WORLD);
    } while (Result);
    free(X_old);
}


int main(int argc, char* argv[]) {
    double Error;
    double start_time, end_time;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (myid == Root) {
        std::ifstream fin("input.txt");
        fin >> MATR_SIZE;
        AB = (double*)malloc(sizeof(double) * MATR_SIZE * (MATR_SIZE + 1));
        for (int j = 0; j < MATR_SIZE * (MATR_SIZE + 1); j++) {
            fin >> AB[j];
        }
        fin.close();
        Error = 1e-5;
    }
    start_time = MPI_Wtime();
    MPI_Bcast(&Error, 1, MPI_DOUBLE, Root, MPI_COMM_WORLD);
    MPI_Bcast(&MATR_SIZE, 1, MPI_INT, Root, MPI_COMM_WORLD);
    X = (double*)malloc(sizeof(double) * MATR_SIZE);
    if (myid == Root) {
        for (int i = 0; i < MATR_SIZE; i++) {
            X[i] = 1;
        }
    }
    MPI_Bcast(X, MATR_SIZE, MPI_DOUBLE, Root, MPI_COMM_WORLD);
    size = (MATR_SIZE / numprocs) + ((MATR_SIZE % numprocs) > myid ? 1 : 0);
    A = (double*)malloc(sizeof(double) * (MATR_SIZE + 1) * size);
    displs = (int*)malloc(numprocs * sizeof(int));
    sendcounts = (int*)malloc(numprocs * sizeof(int));
    SIZE = (MATR_SIZE + 1) * size;
    MPI_Gather(&SIZE, 1, MPI_INT, sendcounts, 1, MPI_INT, Root,
               MPI_COMM_WORLD);
    displs[0] = 0;
    for (int i = 1; i < numprocs; i++) {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
    MPI_Scatterv(AB, sendcounts, displs, MPI_DOUBLE, A,
                 (MATR_SIZE + 1) * size, MPI_DOUBLE, Root, MPI_COMM_WORLD);
    solve(MATR_SIZE, size, Error);
    end_time = MPI_Wtime();
    if (myid == Root) {
        std::cout << "time: " << end_time - start_time << "s ; Error: " << Error << " ; " << numprocs << " processes" << std::endl;
        std::ofstream fout("output.txt");
        for (int i = 0; i < MATR_SIZE; i++) {
            fout << std::fixed << std::setprecision(4);
            fout << X[i] << std::endl;
        }
        fout.close();
    }
    free(sendcounts);
    free(displs);
    free(AB);
    free(A);
    free(X);
    MPI_Finalize();
    return 0;
}


//=============================================
//  time: 0.726254s ; eps = 1e-05 ; 1 processes
//  time: 0.398932s ; eps = 1e-05 ; 2 processes
//  time: 0.338085s ; eps = 1e-05 ; 3 processes
//  time: 0.273432s ; eps = 1e-05 ; 4 processes
//  time: 0.273432s ; eps = 1e-05 ; 5 processes
//  time: 0.242042s ; eps = 1e-05 ; 6 processes
//  time: 0.236082s ; eps = 1e-05 ; 7 processes
//  time: 0.21405 s ; eps = 1e-05 ; 8 processes
//==============================================
