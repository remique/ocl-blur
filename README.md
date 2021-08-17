### OpenCL Gaussian Blur

Gaussian Blur algorithm implemented in OpenCL, written as a learning project. The intention was to benchmark the performance against CPU. However, take the tests below with a grain of salt as CPU calculations were run on a single thread. This shows the power of GPU computing nonetheless.

### Test results

The tests were done on MBP 2017 (Intel Core i5, Intel Iris Plus Graphics 640).

| Kernel size | CPU time | GPU time |
| ----------- | -------- | -------- |
| 5           | 80.038   | 0.033    |
| 10          | 18.610   | 0.012    |
| 20          | 5.600    | 0.005    |

### Sources

- [Image Filters: Gaussian Blur](https://aryamansharda.medium.com/image-filters-gaussian-blur-eb36db6781b1) by Aryaman Sharda
- [GPU Image Processing using OpenCL](https://towardsdatascience.com/get-started-with-gpu-image-processing-15e34b787480) by Harald Scheidl
