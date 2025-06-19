显存问题  
stage_inpaint_pano_greedy_search() 中 segment  

检查坐标系问题  

segment 准确性问题
调试成功  
评价指标  
渲染视频  

结题后：文档  

JS-CHU 想干嘛  



#### 取点补全算法

文本提到了n**类**物体，在全景图中一共识别出m**个**属于这n**类**的物体，这些物体组成一个集合$\{O_i\}_{i=0}^{m-1}$，第 $j$ 类物体一个在文本中有一个描述，记为$prompt[j]$，组成集合$\{prompt_j\}_{j=0}^{n-1}$.

mesh中的点用$vertices: tensor[3,n]$表示，第$i$列表示第$i$个点的坐标.

segmentor里，列表$id2type$把第i个物体映射到第j类，即$id2type[i]=j,j$是第i个物体所属的类别.

对于物体$\ s_i\in\{O_i\}_{i=0}^{m-1}$，它包含的点在vertices中的索引组成索引列表$pidx_i$，即$\{pts_j=vertices[:3][pidx_i[j]]\}_{j=0}^{len(pidx_i)-1}$这些点组成了这个物体$s$. 这个物体的大小$size_i$用点集坐标的标准差的k倍来表征. 即
$$
size_i=k\sigma(\{pts_j\})
$$
其中$\sigma(\{x_i\}_{i=0}^{n-1})$表示标准差运算，即$\sigma(\{x_i\}_{i=0}^{n-1})=\sqrt {\frac{1}{n}\sum_{i=0}^{n-1}(x_i-\bar{x})^2}$, k是比例系数.

如果一个点和物体中心$\bar{pts}$的距离大于$size_i$，可以认为这个点在物体外.

对于每个物体$s_i$，首先，在PERF的circle_camera_sampler取得的相机位置中，我们找到物体外离物体中心$\bar{pts}$最近的位置，作为环绕相机的主视角，生成伴随相机。接着，在伴随相机的里找到视野完整度最小的一个，作为真正的主视角，生成第二份伴随相机。最后，将这份伴随相机按视野完整度从大到小的顺序开始补全，补全时利用$prompt[id_2type[i]]$.



