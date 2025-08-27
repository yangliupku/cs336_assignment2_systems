# CS336 Assignment 2: Systems and Parallelism Report

## Model Specifications

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
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
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
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
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
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
    <tr>
      <td>xl</td>
      <td>1600</td>
      <td>6400</td>
      <td>48</td>
      <td>25</td>
      <td>1.98B</td>
      <td>3157.1ms</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
    <tr>
      <td>2.7B</td>
      <td>2560</td>
      <td>10240</td>
      <td>32</td>
      <td>32</td>
      <td>3.38B</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
  </tbody>
</table>

_Table 1: Model specifications with forward time and parameter count columns added. Forward time and parameter counts to be filled in during benchmarking._
