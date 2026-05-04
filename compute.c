#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

__global__ void computeKernel(vector3 *pos, vector3 *vel, double *mass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUMENTITIES) return;

    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    for (int j = 0; j < NUMENTITIES; j++) {
        if (i != j) {
            double dx = pos[i][0] - pos[j][0];
            double dy = pos[i][1] - pos[j][1];
            double dz = pos[i][2] - pos[j][2];

            double magnitude_sq = dx * dx + dy * dy + dz * dz;
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -1.0 * GRAV_CONSTANT * mass[j] / magnitude_sq;

            ax += accelmag * dx / magnitude;
            ay += accelmag * dy / magnitude;
            az += accelmag * dz / magnitude;
        }
    }

    vel[i][0] += ax * INTERVAL;
    vel[i][1] += ay * INTERVAL;
    vel[i][2] += az * INTERVAL;

    pos[i][0] += vel[i][0] * INTERVAL;
    pos[i][1] += vel[i][1] * INTERVAL;
    pos[i][2] += vel[i][2] * INTERVAL;
}

void compute() {
    static vector3 *dPos = NULL;
    static vector3 *dVel = NULL;
    static double *dMass = NULL;
    static int initialized = 0;

    size_t vecSize = sizeof(vector3) * NUMENTITIES;
    size_t massSize = sizeof(double) * NUMENTITIES;

    if (!initialized) {
        cudaMalloc(&dPos, vecSize);
        cudaMalloc(&dVel, vecSize);
        cudaMalloc(&dMass, massSize);

        cudaMemcpy(dMass, mass, massSize, cudaMemcpyHostToDevice);

        initialized = 1;
    }

    cudaMemcpy(dPos, hPos, vecSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dVel, hVel, vecSize, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (NUMENTITIES + threads - 1) / threads;

    computeKernel<<<blocks, threads>>>(dPos, dVel, dMass);
    cudaDeviceSynchronize();

    cudaMemcpy(hPos, dPos, vecSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dVel, vecSize, cudaMemcpyDeviceToHost);
}
