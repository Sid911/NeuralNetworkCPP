# NeuralNetworkCPP
A small experiment to learn about neural networks and their runtimes in cpp/cuda.

### Logs

- **Jun 23, 2023** : Got basic neural networks working with backpropagation. Still, a lot of bugs to be tested and fixed but `Standard` Dense NN's do work for simple relations like XOR and linear equations like $`(x*2) + 2`$ example:

  https://github.com/Sid911/NeuralNetworkCPP/blob/9ff476fba1769c663bbdb554bd40f276246d896b/out.txt#L401-L411
  As you can see the output is still not converging in many cases and sometimes it just freaks out and returns ridiculous answers. There is a lot to do though but progress nonetheless.

  Also, I need to figure out how to disable Eigen's Cuda compilation for now as this results in a huge increase in build times plus it shows very unnecessary build times warnings as all the functions are prefixed with `__host__` and `__device__` attributes. I am ignoring these errors for now as GPU support is yet to be backed in but this will cause problems in the future as nvcc won't be able to give me warnings.
  https://github.com/Sid911/NeuralNetworkCPP/blob/9ff476fba1769c663bbdb554bd40f276246d896b/CMakeLists.txt#L9

- **June 28, 2023**: Well I think I lied about Xor, I had OR and AND working but I thought XOR would work too. Of course, no it didn't I had a lot of logs stating the problems in obsidian but in short
  - **June 26, 2023**
    - 3 AM - Wrong equation interpretation from 3b1b video _(I am not sure if still all are correct)_
    - 11 PM - wrong activations being saved 😭 caused a lot of pain https://github.com/Sid911/NeuralNetworkCPP/blob/f3d627fe5f07129eed664cb8ac6cdc658c464273/NN/Layers/NNDenseLayer.cu#L67
  - **Today** : I somehow got sigmoid derivative function incorrect, that was embarrasing. https://github.com/Sid911/NeuralNetworkCPP/blob/2e9809c5519e81d6a4ce3187f019de48803dacda/NN/Layers/NNDenseLayer.cuh#L47-L49
  
  Either way I got sigmoid working and relu for xor but they work onyl 30 percent of the time this has something to do with local minima or ... if I am gussing my code. I also got pretty printing now with   ANSI outputs, it looks cool though.
  ![image](https://github.com/Sid911/NeuralNetworkCPP/assets/27860105/7ea901b4-2f2d-46d2-bb53-5f576819267d)
