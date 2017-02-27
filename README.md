WrapCudaDeformer
=====

Custom deformer node. Autodesk® Maya® Plugin.

![ScreenShot](http://mishurov.000webhostapp.com/github/wrap_cuda/wrap.png)

## Warning
CUDA acceleration is <b>not implemented</b> yet!<br/>
It's the simplest implementation of the algorithm described in the paper from Alias|wavefront "Skinning Characters using Surface-Oriented Free-Form Deformations" even without implementing features described in "Extending the Algorithm" section. It's just an intial version. <b>Boost.Geometry is required</b> to calculate distances from points to control elements.

### Todos
<ul>
	<li>Extend the algorithm using paper mentioned above to make it more intuitive for an artist.</li>
	<li>Refactore code to use harmonic coordinates instead of mean value coordinates using Pixar's paper "Harmonic Coordinates for Character Articulation".</li>
	<li>Use CUDA and cuBLAS acceleration to do faster calculations</li>
</ul>


