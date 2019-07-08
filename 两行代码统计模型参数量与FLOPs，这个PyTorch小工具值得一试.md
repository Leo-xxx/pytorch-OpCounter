## 两行代码统计模型参数量与FLOPs，这个PyTorch小工具值得一试

机器之心 [CVer](javascript:void(0);) *昨天*

点击上方“**CVer**”，选择加"星标"或“置顶”

重磅干货，第一时间送达![img](https://mmbiz.qpic.cn/mmbiz_jpg/ow6przZuPIENb0m5iawutIf90N2Ub3dcPuP2KXHJvaR1Fv2FnicTuOy3KcHuIEJbd9lUyOibeXqW8tEhoJGL98qOw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

作者：思源

本文转载自：机器之心

> 你的模型到底有多少参数，每秒的浮点运算到底有多少，这些你都知道吗？近日，GitHub 开源了一个小工具，它可以统计 PyTorch 模型的参数量与每秒浮点运算数（FLOPs）。有了这两种信息，模型大小控制也就更合理了。

其实模型的参数量好算，但浮点运算数并不好确定，我们一般也就根据参数量直接估计计算量了。但是像卷积之类的运算，它的参数量比较小，但是运算量非常大，它是一种计算密集型的操作。反观全连接结构，它的参数量非常多，但运算量并没有显得那么大。



此外，机器学习还有很多结构没有参数但存在计算，例如[最大池化](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247490221&idx=2&sn=5161fe61e5833511ab75efa40f1d8dc3&chksm=f9a26822ced5e1342328cff4aba3fc8fea79339d4ccf49d246be25cfbcd624485a0c54fe9535&mpshare=1&scene=1&srcid=0708rSArXxfime3GZiHG5uPq&key=cfad420b0c7e89f9da5460a9a5b240965caeef55c7ce0fe1f44afe528078204877cd27cc4d86bbe85a4287a3808b336e318c7507f321927672daa9b2fb48a5103840ed1a4002abb3dfc0d71a6444c370&ascene=1&uin=MjMzNDA2ODYyNQ%3D%3D&devicetype=Windows+10&version=62060833&lang=zh_CN&pass_ticket=TJZ2x%2BCeLcNXILoA2fzlvgGCucD2AExSAq4kcuqUm1grb%2BD3%2FE%2FG0iYjqRlvhdTC)和 [Dropout](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247490221&idx=2&sn=5161fe61e5833511ab75efa40f1d8dc3&chksm=f9a26822ced5e1342328cff4aba3fc8fea79339d4ccf49d246be25cfbcd624485a0c54fe9535&mpshare=1&scene=1&srcid=0708rSArXxfime3GZiHG5uPq&key=cfad420b0c7e89f9da5460a9a5b240965caeef55c7ce0fe1f44afe528078204877cd27cc4d86bbe85a4287a3808b336e318c7507f321927672daa9b2fb48a5103840ed1a4002abb3dfc0d71a6444c370&ascene=1&uin=MjMzNDA2ODYyNQ%3D%3D&devicetype=Windows+10&version=62060833&lang=zh_CN&pass_ticket=TJZ2x%2BCeLcNXILoA2fzlvgGCucD2AExSAq4kcuqUm1grb%2BD3%2FE%2FG0iYjqRlvhdTC) 等。因此，PyTorch-OpCounter 这种能直接统计 FLOPs 的工具还是非常有吸引力的。



- PyTorch-OpCounter GitHub 地址：https://github.com/Lyken17/pytorch-OpCounter



**OpCouter**



PyTorch-OpCounter 的安装和使用都非常简单，并且还能定制化统计规则，因此那些特殊的运算也能自定义地统计进去。



我们可以使用 pip 简单地完成安装：pip install thop。不过 GitHub 上的代码总是最新的，因此也可以从 GitHub 上的脚本安装。



对于 torchvision 中自带的模型，Flops 统计通过以下几行代码就能完成：

- 
- 
- 
- 
- 
- 

```
from torchvision.models import resnet50from thop import profilemodel = resnet50()input = torch.randn(1, 3, 224, 224)flops, params = profile(model, inputs=(input, ))
```



我们测试了一下 DenseNet-121，用 OpCouter 统计了参数量与运算量。API 的输出如下所示，它会告诉我们具体统计了哪些结构，它们的配置又是什么样的。



![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWibuP8xYyiaOamPofYpliaYUfDuZSf5zbsZneibsKR1NgQ7foIy0uYZjMsjBorz4YJ0yibtGZPama2CTeQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



最后输出的浮点运算数和参数量分别为如下所示，换算一下就能知道 DenseNet-121 的参数量约有 798 万，计算量约有 2.91 GFLOPs。

- 
- 

```
flops: 2914598912.0parameters: 7978856.0
```



**OpCouter 是怎么算的**



我们可能会疑惑，OpCouter 到底是怎么统计的浮点运算数。其实它的统计代码在项目中也非常可读，从代码上看，目前该工具主要统计了视觉方面的运算，包括各种卷积、激活函数、池化、批归一化等。例如最常见的二维卷积运算，它的统计代码如下所示：

- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 

```
def count_conv2d(m, x, y):    x = x[0]    cin = m.in_channels    cout = m.out_channels    kh, kw = m.kernel_size    batch_size = x.size()[0]    out_h = y.size(2)    out_w = y.size(3)    # ops per output element    # kernel_mul = kh * kw * cin    # kernel_add = kh * kw * cin - 1    kernel_ops = multiply_adds * kh * kw    bias_ops = 1 if m.bias is not None else 0    ops_per_element = kernel_ops + bias_ops    # total ops    # num_out_elements = y.numel()    output_elements = batch_size * out_w * out_h * cout    total_ops = output_elements * ops_per_element * cin // m.groups    m.total_ops = torch.Tensor([int(total_ops)])
```



总体而言，模型会计算每一个卷积核发生的乘加运算数，再推广到整个卷积层级的总乘加运算数。



**定制你的运算统计**



有一些运算统计还没有加进去，如果我们知道该怎样算，那么就可以写个自定义函数。

- 
- 
- 
- 
- 
- 
- 
- 

```
class YourModule(nn.Module):    # your definitiondef count_your_model(model, x, y):    # your rule hereinput = torch.randn(1, 3, 224, 224)flops, params = profile(model, inputs=(input, ),                        custom_ops={YourModule: count_your_model})
```



最后，作者利用这个工具统计了各种流行视觉模型的参数量与 FLOPs 量：



![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWibuP8xYyiaOamPofYpliaYUfDJAKn4KO15VaE988Xt3oAMX0MQfL7TrpAwm0PgIBwErShgPM5Bp11Vg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



**CVer学术交流群**



扫码添加CVer助手，可申请加入**CVer-目标检测交流群、图像分割、目标跟踪、人脸检测&识别、OCR、超分辨率、SLAM、医疗影像、Re-ID、GAN、NAS、深度估计、自动驾驶和剪枝&压缩等群。****一定要备注：****研究方向+地点+学校/公司+昵称**（如目标检测+上海+上交+卡卡）

![img](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oX7pdpBKibicSnmb8wRGicbT0Rhr61k0f922lbXcowibk5DTRibROvFB1yMCAZQvj1iaEe6Qsia9bU0UMJCA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

▲长按加群



![img](https://mmbiz.qpic.cn/mmbiz_png/e1jmIzRpwWg3jTWCAZ4BrnvIuN20lLkhIjtg4GRSDhTk9NpeF0GGTJwUpKPatscIQU7Ndj9hgl8BPpGj2BJoFw/640?tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

▲长按关注我们

**麻烦给我一个在看****！**

[阅读原文](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247490221&idx=2&sn=5161fe61e5833511ab75efa40f1d8dc3&chksm=f9a26822ced5e1342328cff4aba3fc8fea79339d4ccf49d246be25cfbcd624485a0c54fe9535&mpshare=1&scene=1&srcid=0708rSArXxfime3GZiHG5uPq&key=cfad420b0c7e89f9da5460a9a5b240965caeef55c7ce0fe1f44afe528078204877cd27cc4d86bbe85a4287a3808b336e318c7507f321927672daa9b2fb48a5103840ed1a4002abb3dfc0d71a6444c370&ascene=1&uin=MjMzNDA2ODYyNQ%3D%3D&devicetype=Windows+10&version=62060833&lang=zh_CN&pass_ticket=TJZ2x%2BCeLcNXILoA2fzlvgGCucD2AExSAq4kcuqUm1grb%2BD3%2FE%2FG0iYjqRlvhdTC##)



![img](https://mp.weixin.qq.com/mp/qrcode?scene=10000004&size=102&__biz=MzUxNjcxMjQxNg==&mid=2247490221&idx=2&sn=5161fe61e5833511ab75efa40f1d8dc3&send_time=)

微信扫一扫
关注该公众号