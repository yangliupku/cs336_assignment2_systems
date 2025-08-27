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
