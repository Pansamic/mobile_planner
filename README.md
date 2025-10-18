# mobile planner

This is a simple global planner for mobile robots. This system contains an elevation mapping subsystem and a global path planner subsystem. 

## build

`cmake -B build && cmake --build build`

CMake compile options:


## methodology

### 1. grid map data structure

A $m \times n$ grid map with length $L$ and width $W$ is defined as follows:

$$
\begin{bmatrix}
c_{00} & c_{01} & \cdots & c_{0n} \\
c_{10} & c_{11} & \cdots & c_{1n} \\
\vdots & \vdots & \ddots & \vdots \\
c_{m0} & c_{m1} & \cdots & c_{mn}
\end{bmatrix}
$$

in which $c_{ij}$ is a cell at $(i,j)$ in the map.

The geometric center (origin) lies at the center of the map, not $c_{00}$.

### 2. elevation map

Elevation estimation subsystem requires point cloud frames **aligned with odometry frame**.

#### 2.1. point cloud preprocessing

Divide point cloud by grid according to X-Y location of points.
1. Transform point cloud to map frame.
1. Remove roof points by delete points higher than 2m.
2. Reserve at most `max_top_points_in_grid`(default = 5) points in each grid.

#### 2.2. elevation estimation with uncertainty

For each grid, model the points as Gaussian distribution $N(h, \sigma^2)$.

Fuse current elevation measurement with existing (prior) elevation measurement $N(\hat h^-, \hat \sigma^{2-})$ with one-dimensional Kalman Filter [\[1\]](https://ieeexplore.ieee.org/document/8392399).

$$
\hat h^+ = \frac{\hat h^- \sigma^2 + h \hat \sigma^{2-}}{\sigma^2 + \hat \sigma^{2-}}
$$

$$
\hat \sigma^{2+} = \frac{\sigma^2 \hat \sigma^{2-}}{\sigma^2 + \hat \sigma^{2-}}
$$

#### 2.3. elevation filling

Since the sparse point cloud cannot cover all the cells of map, some cells are missing.
So we fill the missing cells with the minimum elevation of 3x3 window.

#### 2.4. elevation smoothing

Apply Gaussian smoothing to the elevation map.

#### slope map

Slope quantifies the steepness of the terrain at each location—i.e., the magnitude of the gradient of the elevation surface.

**Continuous form**:  
$$
h_{\text{slope}}(x, y) = \left\| \nabla h(x, y) \right\| = \sqrt{ \left( \frac{\partial h}{\partial x} \right)^2 + \left( \frac{\partial h}{\partial y} \right)^2 }
$$

**Discrete approximation** (using finite differences on a regular grid with spacing $\Delta x = \Delta y = d$): 

$$
\frac{\partial h}{\partial x} \approx \frac{h[i, j+1] - h[i, j-1]}{2d}, \quad
\frac{\partial h}{\partial y} \approx \frac{h[i+1, j] - h[i-1, j]}{2d}
$$
$$
h_{\text{slope}}[i, j] = \sqrt{ \left( \frac{h[i, j+1] - h[i, j-1]}{2d} \right)^2 + \left( \frac{h[i+1, j] - h[i-1, j]}{2d} \right)^2 }
$$

#### step height map

Step height measures the maximum vertical discontinuity (e.g., curb, rock edge) within a local neighborhood. It helps detect abrupt changes that may be impassable.

For each cell $(i, j)$ , define a local window $\mathcal{W}_{i,j}$ (default 3×3). Then:

$$
h_{\text{step}}[i, j] = \max_{(m,n) \in \mathcal{W}_{i,j}} h[m,n] - \min_{(m,n) \in \mathcal{W}_{i,j}} h[m,n]
$$

Alternatively, to be more sensitive to **local obstacles** (not just overall range), you can compute the maximum absolute difference between the center cell and its neighbors:

$$
h_{\text{step}}[i, j] = \max_{(m,n) \in \mathcal{N}(i,j)} \left| h[m,n] - h[i,j] \right|
$$

where $\mathcal{N}(i,j)$ is the set of immediate neighbors (e.g., 4- or 8-connected).

#### roughness map

$$
h_{\text{rough}}[i, j] = \sqrt{ \frac{1}{|\mathcal{W}|} \sum_{(m,n) \in \mathcal{W}_{i,j}} \left( h[m,n] - \bar{h}_{i,j} \right)^2 }
$$

where $\bar{h}_{i,j}$ is the mean elevation in the window $\mathcal{W}_{i,j}$.

#### traversability map

We can define traversability map $\mathcal M_\tau$ as the combination of slope map $\mathcal M_\Delta$, step height map $\mathcal M_\zeta$, and roughness map $\mathcal M_r$ [\[2\]](https://ieeexplore.ieee.org/document/10610106).

$$
\mathcal M_\tau = \omega_1\frac{\mathcal M_\Delta}{s_{\text{crit}}} + \omega_2\frac{\mathcal M_\zeta}{\zeta_{\text{crit}}} + \omega_3\frac{\mathcal M_r}{r_{\text{crit}}}
$$

where $\omega_1$, $\omega_2$, and $\omega_3$ are weights for slope, step height, and roughness, respectively, totaling to 1. $s_{\text{crit}}$, $\zeta_{\text{crit}}$, and $r_{\text{crit}}$ denote robot-specific critical thresholds for maximum slope, step height, and roughness tolerance before reaching unsafe conditions [\[2\]](https://ieeexplore.ieee.org/document/10610106).
### reachability check

### path planning

### reference

```
@ARTICLE{1,
  author={Fankhauser, Péter and Bloesch, Michael and Hutter, Marco},
  journal={IEEE Robotics and Automation Letters}, 
  title={Probabilistic Terrain Mapping for Mobile Robots With Uncertain Localization}, 
  year={2018},
  volume={3},
  number={4},
  pages={3019-3026},
  keywords={Robot sensing systems;Uncertainty;Legged locomotion;Position measurement;Probabilistic logic;Mapping;field robots;legged robots},
  doi={10.1109/LRA.2018.2849506}
}
@INPROCEEDINGS{2,
  author={Leininger, Abe and Ali, Mahmoud and Jardali, Hassan and Liu, Lantao},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Gaussian Process-based Traversability Analysis for Terrain Mapless Navigation}, 
  year={2024},
  volume={},
  number={},
  pages={10925-10931},
  keywords={Uncertainty;Navigation;Source coding;Vegetation;Gaussian processes;Planning;Vehicle dynamics;Off-road navigation;Traversability-analysis;Gaussian process (GP)},
  doi={10.1109/ICRA57147.2024.10610106}
}

```