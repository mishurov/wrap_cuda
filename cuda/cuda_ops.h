#ifndef MAYA_CUDA_OPS_H_
#define MAYA_CUDA_OPS_H_

#include <stdio.h>
#include <sys/time.h>

int CudaDeviceCount();

void CudaComputeControlPoints(double *P, unsigned int *Ridx, double *R,
			double *C, unsigned int numRefs,
			unsigned int numPoints, unsigned int numElems);

void CudaComputeWeights(float local, double *D, double *Wnorm,
			unsigned int numPoints, unsigned int numElems);

void CudaApplyDeform(double *C, double *Wnorm, unsigned int *DRidx,
		double *DR, double *P, unsigned int numDriv,
		unsigned int numPoints, unsigned int numElems);


#endif  // MAYA_CUDA_OPS_H_
