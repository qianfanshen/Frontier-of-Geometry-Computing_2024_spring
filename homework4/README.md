# 几何计算前沿 第四次作业

DDL：2024.6.16

### 任务

本次作业需要完成一个基于MLP的从点云重建表面的模型，训练数据是表面上的点云、空间中的采样点和对应SDF值，输出重建后的形状。

MLP的功能是输出空间中任意查询点的SDF值。假设输入的点云为$P$，查询点为$q \in Q$，则有：
$$
\tilde{F}(q) = \text{MLP}(q)
$$
训练时的损失函数为：
$$
L_{sdf} = \sum_{q_i \in Q} \| \tilde{F}(q_i) - F(q_i) \|^2 + \| \nabla \tilde{F}(q_i) - N(q_i) \|^2
$$
其中$\tilde{F}(q_i)$为MLP预测的SDF值，$F(q_i)$为ground truth的SDF值，$\nabla \tilde{F}(q_i)$为SDF相对$q_i$的梯度（可以通过`torch.autograd.grad`计算），$N(q_i)$为ground truth的梯度。

测试时在空间中均匀采样，计算每个采样点的SDF值，然 后使用Marching Cubes算法提取mesh。

### 数据

数据在data文件夹下，为修复后的ShapeNet V1中的airplane类。

每个uid文件夹下包括：

```
pointcloud.npz
- "points": 表面上的点云坐标
- "normals": 表面上的法向量
sdf.npz
- "points": 空间中的采样点坐标
- "grad": 采样点对应梯度
- "sdf": 采样点对应SDF值
*.obj: 用于参考的ground truth三维形状
```

### 拓展

MLP会更倾向于生成光滑的结果，使得重建表面的细节不足。

实现一个基于Fourier feature的position encoding，将$q$的坐标进行Fourier特征映射后再输入MLP，比较和原方法的结果。

> 参考论文：Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains

### 评分标准

代码实现：5分

重建结果：5分

报告：3分

拓展：2分

