# NeuralNetworkCPP
A small experiment to learn about neural networks and their runtimes in cpp/cuda.

### Logs

- **Jun 23, 2023** : Got baisc neural networks working with backpropagation. Still a lot of bugs to be tested and fixed but `Standard` Dense NN's do work for simple relations like XOR and linear equations like $`(x*2) + 2`$ example:

  https://github.com/Sid911/NeuralNetworkCPP/blob/9ff476fba1769c663bbdb554bd40f276246d896b/out.txt#L401-L411
  As you can see the output is still not converging in many cases and sometimes it just freaks out and returns ridiculous answers. There is a lot to do though but progress nonetheless.

  Also, I need to figure out how to disable Eigen's Cuda compilation for now as this results in a huge increase in build times plus it shows very unnecessary build times warnings as all the functions are prefixed with `__host__` and `__device__` attributes. I am ignoring these errors for now as GPU support is yet to be backed in but this will cause problems in the future as nvcc won't be able to give me warnings.
  https://github.com/Sid911/NeuralNetworkCPP/blob/9ff476fba1769c663bbdb554bd40f276246d896b/CMakeLists.txt#L9
