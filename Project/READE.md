# 几何计算前沿 大作业

## 作业说明：
本次作业实现了有交互功能的网格变形算法、网格平滑算法、网格简化算法以及网格参数化算法。其中UI界面主要依赖于Open3D，函数功能主要依赖于numpy与Trimesh库。

## 使用方法：
### Menu
点击左上角File目录，选择使用的Mesh模型（.obj文件）。

点击左下角的Reset Mesh按钮还原网格。

### Settings
在右侧的Setting栏选择Mode，可以选择网格变形（Deformation）, 网格平滑（Smoothing）, 网格简化（Simplify）, 网格参数化（Parameterization）以及观察（Check）模式，其中Check模式用于观察操作后的三维物体，不会对网格进行操作。

在右侧选择Setting栏选择Shader，用于给Mesh着色。有defaultLit, defaultUnlit, normals, unlitLine四种着色器选择。

可以在Output Filename栏填写处理后网格导出的文件名，所有导出的网格都是.obj格式，均位于"./models"文件夹目录下，默认为"output.obj"，在此栏填写只用填写导出的文件名（如"output.obj"只用填写output）。

### Deformation：
滑动Nearest Points Num选择handle点附近最近的x个点作为Deformation算法的输入点云。

滑动Iteration选择变形迭代数。

按住CTRL键加鼠标左键选择handle点。

按住CTRL键加鼠标右键取消选择handle点。

按住SHIFT键加鼠标左键并拖拽给定网格变形的具体方向（注意可能滑动到远平面，会输出WARNING）

### Smoothing：
可以选择显示（explicit）或隐式（implicit）方法进行网格平滑。

滑动Iteration选择平滑迭代数。

滑动Learning Rate选择变化学习率。

按住CTRL键加鼠标左键开始Smoothing。

### Simplify：
滑动Ratio选择简化后剩余顶点数占原顶点数的比例。

按住CTRL键加鼠标左键开始Simplify。

### Parameterization：
按住CTRL键加鼠标左键开始Parameterization。

### Check：
按住鼠标左键拖拽对物体进行旋转。

按住CTRL键加鼠标左键进行拖拽，改变物体位置。

按住SHIFT键并滚动鼠标轴进行缩放。