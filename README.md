WrapCudaDeformer
=====

A Proof of Concept deformer node, Autodesk® Maya® Plugin.

![ScreenShot](http://mishurov.usite.pro/github/wrap_cuda/wrap.png)

## Warning
The plug-in isn't intended for production use, it's highly unstable: crashes on heavy geometry, doesn't handle exceptions, uses blocked ranges, etc.

### Info
The CPU part is multithreaded using Intel TBB, the GPU part is mostly implemented with custom kernels and uses the cuBLAS batched subroutine for matrix inversion. There're several ways to make it faster, e.g., to use more intensively cuBLAS so as to fully utilise hardware, to use more clever parallel reduction functions and whatnot.

The deformer implements the basic algorithm described in the paper<br>
<b>Alias|wavefront "Skinning Characters using Surface-Oriented Free-Form Deformations"</b><br>
The algorithm for finding a distance from a point to a triangle is from the document made by Geometric Tools, LLC<br>
https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf

### Asymptotics
The testing data was quite small, nonetheless it gave some insights. At first glance, the GPU mode doesn't perform well in comparison to its CPU counterpart.   

![ScreenShot](http://mishurov.usite.pro/github/wrap_cuda/total.png)

However, it seems that the bottleneck is the memory allocations on a GPU and the data transfers between a GPU and a CPU. The actual computations on a GPU increase with the noticeable lower rate than on a CPU. In the current implementation the CUDA part allocates, transfers and frees all necessary data on every call during the deformation. Static data storage for pointers to device memory or using Unified Memory can solve the problem. 

![ScreenShot](http://mishurov.usite.pro/github/wrap_cuda/separate.png)


The repository contains a csv file with the benchmarks and a simple R script for plotting the data.

