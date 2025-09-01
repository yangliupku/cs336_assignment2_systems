# CS336 Assignment 2: Systems and Parallelism Report

## Problem (benchmarking_script)

<table>
  <thead>
    <tr>
      <th rowspan="2">Size</th>
      <th rowspan="2">d_model</th>
      <th rowspan="2">d_ff</th>
      <th rowspan="2">num_layers</th>
      <th rowspan="2">num_heads</th>
      <th rowspan="2">num_params</th>
      <th colspan="2">CPU</th>
      <th colspan="2">MPS</th>
      <th colspan="2">CUDA</th>
    </tr>
    <tr>
      <th>forward_time</th>
      <th>forward+backward</th>
      <th>forward_time</th>
      <th>forward+backward</th>
      <th>forward_time</th>
      <th>forward+backward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>small</td>
      <td>768</td>
      <td>3072</td>
      <td>12</td>
      <td>12</td>
      <td>120.9M</td>
      <td>250.7ms</td>
      <td>965.8ms</td>
      <td>38.4ms</td>
      <td>158.0ms</td>
      <td>24.2ms</td>
      <td>80.4ms</td>
    </tr>
    <tr>
      <td>medium</td>
      <td>1024</td>
      <td>4096</td>
      <td>24</td>
      <td>16</td>
      <td>412.9M</td>
      <td>816.0ms</td>
      <td>3186.2ms</td>
      <td>119.6ms</td>
      <td>498.8ms</td>
      <td>64.4ms</td>
      <td>243.7ms</td>
    </tr>
    <tr>
      <td>large</td>
      <td>1280</td>
      <td>5120</td>
      <td>36</td>
      <td>20</td>
      <td>956.6M</td>
      <td>1639.8ms</td>
      <td>6740.9ms</td>
      <td>255.8ms</td>
      <td>1110.4ms</td>
      <td>139.9ms</td>
      <td>520.8ms</td>
    </tr>
    <tr>
      <td>xl</td>
      <td>1600</td>
      <td>6400</td>
      <td>48</td>
      <td>25</td>
      <td>1.98B</td>
      <td>3157.1ms</td>
      <td>19000.5ms</td>
      <td>518.5ms</td>
      <td></td>
      <td>285.7ms</td>
      <td>1069.6msa</td>
    </tr>
    <tr>
      <td>2.7B</td>
      <td>2560</td>
      <td>10240</td>
      <td>32</td>
      <td>32</td>
      <td>3.38B</td>
      <td>4685.3ms</td>
      <td>81446.5ms</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>421.8ms</td>
      <td>1616.5ms</td>
    </tr>
  </tbody>
</table>

_Table 1: Model specifications with forward time and parameter._

### effect of warmup.

In the case of `large` model, with CUDA:

_with warmup_:

```
forward [139.71225545 139.52491432 139.59536701 139.69031721 139.49538767
 139.58601654 139.55771551 139.51878622 139.61970806 139.56166804]
forward-backward [520.85475251 520.93084529 519.79393885 520.96752822 521.21670544
 520.71822435 520.59522644 520.83975449 520.8985284  592.31639281]
```

_without warmup_:

```
forward [1763.26469705  139.6474801   139.6288909   139.38480616  139.86116648
  139.32373002  139.37116787  139.57205787  139.32672143  139.6548599 ]
forward-backward [922.03254253 519.65327188 539.19908032 519.58310977 519.51987296
 519.59372684 519.47452128 519.45736259 519.44212988 519.27106827]
```

we can see without warmup, the first and third run with backward takes longer than normal. This behavior is repeatable.

### Does context length change model size?

No

```
-------context length = 256---------
total_params (M) 120.945408


-------context length = 1024---------
total_params (M) 120.945408
```

## Profiling

### PyTorch Profiler Results for Large with forward

| Name                                  | Self CPU % | Self CPU  | CPU total % | CPU total | CPU time avg | Self CUDA | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
| ------------------------------------- | ---------- | --------- | ----------- | --------- | ------------ | --------- | ----------- | ---------- | ------------- | ---------- |
| aten::einsum                          | 2.65%      | 3.264ms   | 19.43%      | 23.940ms  | 73.661us     | 0.000us   | 0.00%       | 123.479ms  | 379.934us     | 325        |
| aten::bmm                             | 3.70%      | 4.560ms   | 6.23%       | 7.680ms   | 23.631us     | 118.812ms | 86.07%      | 122.976ms  | 378.388us     | 325        |
| ampere_sgemm_128x64_tn                | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 83.281ms  | 60.33%      | 83.281ms   | 383.783us     | 217        |
| ampere_sgemm_32x128_tn                | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 29.878ms  | 21.64%      | 29.878ms   | 829.938us     | 36         |
| aten::mul                             | 2.56%      | 3.159ms   | 4.33%       | 5.338ms   | 10.549us     | 6.847ms   | 4.96%       | 6.885ms    | 13.606us      | 506        |
| cudaLaunchKernel                      | 7.07%      | 8.717ms   | 14.80%      | 18.237ms  | 8.853us      | 0.000us   | 0.00%       | 4.689ms    | 2.276us       | 2060       |
| Unrecognized                          | 7.73%      | 9.520ms   | 7.73%       | 9.520ms   | 226.660us    | 4.689ms   | 3.40%       | 4.689ms    | 111.652us     | 42         |
| elementwise_kernel<128, 2, ...>       | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 4.232ms   | 3.07%       | 4.232ms    | 9.750us       | 434        |
| ampere_sgemm_128x128_nn               | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 3.734ms   | 2.71%       | 3.734ms    | 103.734us     | 36         |
| vectorized_elementwise_kernel<4, ...> | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 2.616ms   | 1.89%       | 2.616ms    | 36.329us      | 72         |
| aten::where                           | 0.32%      | 394.998us | 1.32%       | 1.622ms   | 22.524us     | 1.160ms   | 0.84%       | 2.384ms    | 33.113us      | 72         |
| aten::div                             | 0.43%      | 532.896us | 0.75%       | 926.758us | 12.872us     | 2.093ms   | 1.52%       | 2.225ms    | 30.899us      | 72         |
| elementwise_kernel<128, 2, ...>       | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 1.917ms   | 1.39%       | 1.917ms    | 8.876us       | 216        |
| ampere_sgemm_128x128_tn               | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 1.698ms   | 1.23%       | 1.698ms    | 47.155us      | 36         |
| aten::sub                             | 0.54%      | 661.797us | 0.82%       | 1.015ms   | 9.396us      | 1.288ms   | 0.93%       | 1.288ms    | 11.927us      | 108        |
| elementwise_kernel<128, 2, ...>       | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 1.190ms   | 0.86%       | 1.190ms    | 33.063us      | 36         |
| vectorized_elementwise_kernel<4, ...> | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 1.184ms   | 0.86%       | 1.184ms    | 5.480us       | 216        |
| elementwise_kernel<128, 2, ...>       | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 1.160ms   | 0.84%       | 1.160ms    | 32.216us      | 36         |
| aten::concatenate                     | 0.04%      | 54.514us  | 2.01%       | 2.478ms   | 34.418us     | 0.000us   | 0.00%       | 1.156ms    | 16.062us      | 72         |
| aten::cat                             | 1.16%      | 1.433ms   | 1.97%       | 2.424ms   | 33.661us     | 1.156ms   | 0.84%       | 1.156ms    | 16.062us      | 72         |
| aten::add                             | 1.06%      | 1.300ms   | 1.66%       | 2.045ms   | 9.422us      | 1.022ms   | 0.74%       | 1.022ms    | 4.708us       | 217        |
| aten::max                             | 0.32%      | 390.292us | 0.42%       | 520.597us | 14.461us     | 984.770us | 0.71%       | 984.770us  | 27.355us      | 36         |
| reduce_kernel<512, 1, ...>            | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 984.770us | 0.71%       | 984.770us  | 27.355us      | 36         |
| elementwise_kernel<128, 2, ...>       | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 982.851us | 0.71%       | 982.851us  | 27.301us      | 36         |
| vectorized_elementwise_kernel<4, ...> | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 902.497us | 0.65%       | 902.497us  | 25.069us      | 36         |

#### Summary

- **Self CPU time total:** 123.209ms
- **Self CUDA time total:** 138.042ms

### PyTorch Profiler Results for Large with forward and backward

| Name                                                 | Self CPU % | Self CPU  | CPU total % | CPU total | CPU time avg | Self CUDA | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
| ---------------------------------------------------- | ---------- | --------- | ----------- | --------- | ------------ | --------- | ----------- | ---------- | ------------- | ---------- |
| aten::bmm                                            | 1.41%      | 13.378ms  | 47.52%      | 450.567ms | 462.120us    | 356.699ms | 70.02%      | 360.764ms  | 370.014us     | 975        |
| autograd::engine::evaluate_function: BmmBackward0    | 0.20%      | 1.917ms   | 4.40%       | 41.682ms  | 128.252us    | 0.000us   | 0.00%       | 238.175ms  | 732.845us     | 325        |
| BmmBackward0                                         | 0.15%      | 1.436ms   | 4.19%       | 39.765ms  | 122.353us    | 0.000us   | 0.00%       | 238.175ms  | 732.845us     | 325        |
| aten::einsum                                         | 0.36%      | 3.431ms   | 45.30%      | 429.518ms | 1.322ms      | 0.000us   | 0.00%       | 123.077ms  | 378.700us     | 325        |
| Optimizer.step#AdamW.step                            | 1.25%      | 11.823ms  | 15.08%      | 142.990ms | 142.990ms    | 0.000us   | 0.00%       | 116.583ms  | 116.583ms     | 1          |
| Optimizer.step#AdamW.step                            | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 89.233ms  | 17.52%      | 89.233ms   | 89.233ms      | 1          |
| ampere_sgemm_128x64_tn                               | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 83.098ms  | 16.31%      | 83.098ms   | 382.939us     | 217        |
| aten::mul                                            | 1.62%      | 15.402ms  | 10.40%      | 98.571ms  | 27.704us     | 50.138ms  | 9.84%       | 60.563ms   | 17.022us      | 3558       |
| ampere_sgemm_32x32_sliced1x4_nt                      | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 56.182ms  | 11.03%      | 56.182ms   | 780.304us     | 72         |
| ampere_sgemm_128x32_sliced1x4_nn                     | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 55.582ms  | 10.91%      | 55.582ms   | 761.398us     | 73         |
| cudaLaunchKernel                                     | 14.92%     | 141.412ms | 25.17%      | 238.616ms | 21.811us     | 0.000us   | 0.00%       | 37.815ms   | 3.457us       | 10940      |
| Unrecognized                                         | 22.83%     | 216.432ms | 22.83%      | 216.432ms | 95.261us     | 37.815ms  | 7.42%       | 37.815ms   | 16.644us      | 2272       |
| ampere_sgemm_128x128_nn                              | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 34.336ms  | 6.74%       | 34.336ms   | 317.929us     | 108        |
| vectorized_elementwise_kernel<4, CUDAFunctor_add...> | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 33.784ms  | 6.63%       | 33.784ms   | 16.934us      | 1995       |
| vectorized_elementwise_kernel<4, AUnaryFunctor...>   | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 29.808ms  | 5.85%       | 29.808ms   | 14.140us      | 2108       |
| ampere_sgemm_32x128_tn                               | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 29.781ms  | 5.85%       | 29.781ms   | 827.239us     | 36         |
| ampere_sgemm_64x32_sliced1x4_nt                      | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 29.528ms  | 5.80%       | 29.528ms   | 205.053us     | 144        |
| ampere_sgemm_32x128_nt                               | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 28.628ms  | 5.62%       | 28.628ms   | 773.726us     | 37         |
| ampere_sgemm_128x32_nn                               | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 28.137ms  | 5.52%       | 28.137ms   | 195.393us     | 144        |
| aten::add                                            | 0.55%      | 5.211ms   | 2.84%       | 26.886ms  | 21.170us     | 20.185ms  | 3.96%       | 27.672ms   | 21.789us      | 1270       |
| aten::sub\_                                          | 0.35%      | 3.338ms   | 1.89%       | 17.940ms  | 27.430us     | 13.464ms  | 2.64%       | 20.089ms   | 30.717us      | 654        |
| aten::div                                            | 0.54%      | 5.147ms   | 1.54%       | 14.598ms  | 23.622us     | 14.101ms  | 2.77%       | 17.586ms   | 28.456us      | 618        |
| elementwise_kernel<128, 2, ...>                      | 0.00%      | 0.000us   | 0.00%       | 0.000us   | 0.000us      | 16.473ms  | 3.23%       | 16.473ms   | 24.048us      | 685        |
| aten::copy\_                                         | 0.30%      | 2.841ms   | 4.41%       | 41.785ms  | 67.943us     | 15.831ms  | 3.11%       | 15.925ms   | 25.895us      | 615        |
| autograd::engine::evaluate_function: MulBackward0    | 0.31%      | 2.923ms   | 5.09%       | 48.272ms  | 95.399us     | 0.000us   | 0.00%       | 12.732ms   | 25.162us      | 506        |

#### Summary

- **Self CPU time total:** 948.116ms
- **Self CUDA time total:** 509.430ms

## Problem (nsys_profile)

(a): For `large` model with 256 context length, we measured 139.9ms from python benchmarking, and 141.8ms from nsys. results match

(b) When look at the fowards pass, the kernels taking most time are `ampere_sgemm_128x64_tn` and `ampere_sgemm_32x128_tn`.

- These are sgemm (single precision general mat mul) on Ampere (A100).
- they correspond to aten::bmm (batch mat mul)
- they are mostly in FFN (SwiGLU) layers, but also in Atten.

| Time  | Total Time | Instances | Avg        | Med        | Min        | Max        | StdDev     | Name                                                           |
| ----- | ---------- | --------- | ---------- | ---------- | ---------- | ---------- | ---------- | -------------------------------------------------------------- |
| 60.4% | 83.241 ms  | 217       | 383.601 μs | 203.553 μs | 201.889 μs | 1.438 ms   | 258.433 μs | ampere_sgemm_128x64_tn                                         |
| 21.7% | 29.867 ms  | 36        | 829.630 μs | 829.876 μs | 825.828 μs | 833.796 μs | 2.086 μs   | ampere_sgemm_32x128_tn                                         |
| 3.1%  | 4.243 ms   | 434       | 9.775 μs   | 10.592 μs  | 7.328 μs   | 12.384 μs  | 1.657 μs   | elementwise_kernel<128, 2, BinaryFunctor<MulFunctor>...>       |
| 2.7%  | 3.734 ms   | 36        | 103.708 μs | 103.712 μs | 103.329 μs | 104.448 μs | 231 ns     | ampere_sgemm_128x128_nn                                        |
| 1.9%  | 2.623 ms   | 72        | 36.435 μs  | 36.320 μs  | 33.536 μs  | 40.000 μs  | 2.396 μs   | vectorized_elementwise_kernel<4, BinaryFunctor<MulFunctor>...> |
| 1.4%  | 1.921 ms   | 216       | 8.891 μs   | 8.128 μs   | 7.904 μs   | 12.544 μs  | 1.533 μs   | elementwise_kernel<128, 2, direct_copy_kernel_cuda...>         |
| 1.2%  | 1.702 ms   | 36        | 47.290 μs  | 47.328 μs  | 46.816 μs  | 47.776 μs  | 222 ns     | ampere_sgemm_128x128_tn                                        |

in backward mode, matrix multiplication is takes most time, but the size is different
| Time | Total Time | Instances | Avg | Med | Min | Max | StdDev | GridXYZ | BlockXYZ | Name |
|------|------------|-----------|-----|-----|-----|-----|--------|---------|----------|------|
| 19.6% | 56.519 ms | 72 | 784.981 μs | 784.468 μs | 772.548 μs | 793.604 μs | 4.681 μs | 160 40 1 | 128 1 1 | ampere_sgemm_32x32_sliced1x4_nt |
| 18.9% | 54.448 ms | 72 | 756.219 μs | 756.179 μs | 744.388 μs | 765.316 μs | 4.576 μs | 10 32 2 | 256 1 1 | ampere_sgemm_128x32_sliced1x4_nn |
| 10.3% | 29.675 ms | 144 | 206.079 μs | 205.937 μs | 202.273 μs | 208.897 μs | 1.249 μs | 20 40 1 | 256 1 1 | ampere_sgemm_64x32_sliced1x4_nt |
| 9.7% | 28.085 ms | 143 | 196.396 μs | 196.417 μs | 192.833 μs | 199.489 μs | 1.228 μs | 10 32 4 | 256 1 1 | ampere_sgemm_128x32_nn |
| 9.5% | 27.357 ms | 36 | 759.928 μs | 758.372 μs | 748.292 μs | 768.804 μs | 4.810 μs | 40 40 1 | 256 1 1 | ampere_sgemm_32x128_nt |
| 9.4% | 26.987 ms | 36 | 749.650 μs | 749.955 μs | 737.988 μs | 758.724 μs | 4.691 μs | 40 8 3 | 256 1 1 | ampere_sgemm_128x128_nn |

(c): except for matrix multiplication, there's also 7% element wise kernels, being used in RoPE and dot product attention

(d): Combining forward, backward and opt, the elementwise kernels now that about 20% of overall time.

(e) Softmax takes about 36% of overall time in attention, where as matmul takes 53%.
| Name | Start | Duration | TID | GPU | Context |
|------|-------|----------|-----|-----|---------|
| scaled_dot_product_attention [358.017 μs] | 38.5546s | 358.017 μs | 14874 | GPU 0 | Stream 7 |
| computing softmax scores [73.152 μs] | 38.5546s | 73.152 μs | 14874 | GPU 0 | Stream 7 |
| aten::where, seq = 29087, op_id = 233097 [35.296 μs] | 38.5547s | 35.296 μs | 14874 | GPU 0 | Stream 7 |
| computing softmax [129.153 μs] | 38.5547s | 129.153 μs | 14874 | GPU 0 | Stream 7 |
| final matmal [117.056 μs] | 38.5549s | 117.056 μs | 14874 | GPU 0 | Stream 7 |

### benchmark forward and backward pass with mixed precision (float16)

(on A100, with batch size = 4)

| Model Size | Context Length | Forward Time (ms)      |              |                |               | Forward + Backward Time (ms) |              |                |               |
| ---------- | -------------- | ---------------------- | ------------ | -------------- | ------------- | ---------------------------- | ------------ | -------------- | ------------- |
|            |                | **No Mixed Precision** | **bfloat16** | **Difference** | **% Change**  | **No Mixed Precision**       | **bfloat16** | **Difference** | **% Change**  |
| **Small**  | 128            | 29.5                   | 31.7         | +2.2           | +7.5% ⚠️      | 76.6                         | 86.8         | +10.2          | +13.3% ⚠️     |
| **Small**  | 256            | 29.6                   | 32.3         | +2.7           | +9.3% ⚠️      | 85.8                         | 87.4         | +1.6           | +1.9% ⚠️      |
| **Small**  | 512            | 44.6                   | 33.1         | -11.5          | **-25.8%** ✅ | 146.2                        | 94.6         | -51.6          | **-35.3%** ✅ |
| **Small**  | 1024           | 99.8                   | 38.9         | -60.9          | **-61.0%** ✅ | 323.3                        | 126.5        | -196.8         | **-60.9%** ✅ |
| **Medium** | 128            | 62.5                   | 64.1         | +1.6           | +2.6% ⚠️      | 174.9                        | 183.9        | +9.0           | +5.1% ⚠️      |
| **Medium** | 256            | 64.9                   | 66.4         | +1.5           | +2.3% ⚠️      | 234.3                        | 170.5        | -63.8          | **-27.2%** ✅ |
| **Medium** | 512            | 133.5                  | 64.3         | -69.2          | **-51.8%** ✅ | 442.2                        | 186.4        | -255.8         | **-57.9%** ✅ |
| **Medium** | 1024           | 295.4                  | 106.1        | -189.3         | **-64.1%** ✅ | 951.6                        | 351.1        | -600.5         | **-63.1%** ✅ |
| **Large**  | 128            | 88.4                   | 96.6         | +8.2           | +9.3% ⚠️      | 334.3                        | 292.6        | -41.7          | **-12.5%** ✅ |
| **Large**  | 256            | 139.7                  | 96.5         | -43.2          | **-30.9%** ✅ | 513.2                        | 293.5        | -219.7         | **-42.8%** ✅ |
| **Large**  | 512            | 278.2                  | 92.8         | -185.4         | **-66.6%** ✅ | 951.1                        | 343.3        | -607.8         | **-63.9%** ✅ |
| **Large**  | 1024           | 625.4                  | 198.8        | -426.6         | **-68.2%** ✅ | 2034.6                       | 668.5        | -1366.1        | **-67.1%** ✅ |
| **XL**     | 128            | 145.0                  | 133.8        | -11.2          | **-7.7%** ✅  | 616.2                        | 457.6        | -158.6         | **-25.7%** ✅ |
| **XL**     | 256            | 278.0                  | 131.8        | -146.2         | **-52.6%** ✅ | 1043.9                       | 445.1        | -598.8         | **-57.4%** ✅ |
| **XL**     | 512            | 561.5                  | 133.5        | -428.0         | **-76.2%** ✅ | 1885.9                       | 571.7        | -1314.2        | **-69.7%** ✅ |
| **XL**     | 1024           | ❌ Failed              | ❌ Failed    | N/A            | N/A           | ❌ Failed                    | ❌ Failed    | N/A            | N/A           |
