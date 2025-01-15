# Everyday Plan

CTC loss

grad_norm

32x1500帧x1280维度。1500帧是多少秒 （单batch效果会更好？）

音频是连续的频率上的值



<blank> 空白音频

Sense voice





### **Asr模型训练细节**

input: text + audio

output: text

data: 车载音频，小模型的ASR文本，人工标注的ASR文本（label）

 

1. 评估base的纠错能力 CER （涉敏数据用于评测 ）
   1. Base1：小模型ASR 
   2. Base2：小模型ASR+audio 过Qwen2Audio后纠错后的ASR 
   3. Base3:  Qwen2Audio ASR结果 
2. 训练 
   1. 实现模型修改：把模型文件如modeling_qwen2_audio.py, configuration_qwen2_audio.py，在训练文件中直接引入。在训练代码中替换AutoModel，AutoConfig类为具体的类，如Qwen2AudioForConditionalGeneration

```
from qwen2audio_basic.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration
from qwen2audio_basic.processing_qwen2_audio import Qwen2AudioProcessor
```

1. 

2. 加载训练数据时，collate_fn用于处理dataloader读出的batch数据（用processor对batch音频数据进行加载逻辑 放到collate_fn处理，减少全局存储的占用，否则引发cpu memory 不足的报错）

3. qwen2audio源码解析：
   **Processor + AudioEncoder + Projector + LLM**

4. 1. **Processor** includes
      feature_extractor : WhisperFeatureExtractor  （audio to fbank_feature [input_feature])
      tokenizer (text to token [input_ids])
   2. **AudioEncoder** (Whisper Encoder) [对音频特征进行降维：特征提取]
      2 CNN ：conv1d +gelu +conv1d + gelu
      decoder layers (attn + mlp+ norm + dropout)
   3. **LLM** (Qwen-7B)
      MultiModalProjector (音频维度映射到text维度）：Audio embedding to text embedding
      Merge: merge text and audios (After projection) 替换<|AUDIO|>为对应的音频embedding
      （output loss from language model）
   4. 预训练过程中，需同时训练 AudioEncoder + projectord冻结LLM，使音频和文本特征对齐
   5. SFT过程中，只训练LLM (冻结projector+AudioEnocder）

5. 评估训练模型后的纠错能力 CER

   1. 推理：
      1. 纠错 prompt + **input**（小模型ASR+Audio） **output**：（纠错后的ASR）
      2. ASR prompt + input （Audio） output：ASR
   2. 评估：CER（ASR result，人工标注Label）
      1. 线上ASR
      2. 大模型ASR
      3. 大模型纠错ASR



### Today

ASRresult：name+'\t'+asr    因为asr中间可能包含空格 所以以制表符做 分隔符

- [ ] 多种能力数据 训练 （不局限于ASR)



1.11

- [ ] 100h小时结果
- [ ] 提取数据流程
- [ ] 100h 全参训AudioEncoder
- [ ] 质检结果
- [ ] 代码文档整理
- [ ] Lora



1.10

- [x] merge数据处理流程代码整理
- [ ] 数据标注 数据处理脚本
- [ ] tensor board
- [ ] 上传多批数据
- [ ] Lora



1.7

- [x] 录入测试结果
- [x] 确定一下什么原因导致 complete g1 s1 asr不准确
- [x] 你大模型的训练环境和使用方法 写个文档给我吧 To mengxi
- [x] Batch2 
- [ ] 模糊部分：～ 【～】

1.6

- [ ] Tensor board
- [x] 不拼接过渡静音，同样冻结方式的实验
- [ ] 确认audioencoder哪些层进行冻结
- [ ] 确认冻结层之后，增大数据量
- [ ] 尝试p/q-LoRA , 基于风险任务微调llm
- [ ] Langchao 删数据



训后面两层

训后面两层的attention









1.2

- [x] 整理一下结果代码
- [x] asr纠错实验/冻结模型, 测试结果汇总
- [x] Paper reading (asr cor 周会分享)
- [x] 统计一下评测集中是否有模糊字段
- [x] 冻结模型测试
- [x] 重新上传一批之前的数据，方便测cpd

12.30

- [x] asr纠错实验

- [x] Paper reading (asr cor 周会分享)

- [ ] 过度音是否那么有效 (不是那么有效)

- [ ] 训练策略：

  Stage1: 让模型对业务音频有感知能力

  - [x] 1）训 AudioEncoder+Projector
  - [x] 2）训 AudioEncoder
  - [ ] 3）训 部分AudioEncoder层 + Projector

  Stage2: 让模型对风险指令进行follow

  - [ ] 1）全量微调LLM
  - [ ] 2）qlora、lora微调LLM
  - [ ] 3）多种数据训练

  



12.28

- [ ] 选课
- [x] 不输出音频的bad case 检测
- [x] 提交新数据
- [x] ASR纠错的论文和实验设置
- [ ] model train



12.26

- [x] Check 6h的 数据质量
- [x] 用新的评测集去做测试
- [ ] 提交新数据
- [ ] paper reading



12.25

业务数据需求强ASR能力，单是大量业务数据用于训练会影响ASR能力

- [x] 行程数据长啥样 有啥特点 以及要做的任务:  车内噪音、方言、日常对话 
- [x] 提交一批新数据
- [ ] 音频拼接：audio拼接 静音1s ， asr拼接 **空格还是句号**？
- [ ] 代码check：loss计算，loss曲线
- [ ] whisper AudioEncoder和featureextractor有对30s做什么限制：embeeding维度的限制？：featureextractor 限制在30s的音频截断或填充
- [x] 实验：30s以上的音频参与训练是否会提升模型性能, 1s以下音频是否会影响模型性能



- [ ] ASR多种数据微调: 
  - [ ] 清晰的ASR数据 (高质量数据训练）+ 多种音频数据（数据多样性）
  - [ ] 数据配比
  - [ ] 噪音+方言中文音频
  - [x] Wenet 5h 数据 拼接后 + merge 30s + car noise 背景音
  - [ ] 按任务对音频数据做划分
  - [ ] 同一个音频 + noise 的音频特征差别



- [ ] Qwen2Audio的架构、微调（encoder+processor+llm） : CLIP + VIT + LLAVA 1-1.5 
- [ ] whisper & 其他open source的slm微调方案
- [ ] 多种数据微调
- [x] 冻结部分
- [x] 单一数据训练 data scale 到什么地步不会过拟合 6h 10h 20h 50h 
- [ ] 分批上传标注好的文档

- [ ] 代码整理

Insight:

- 数据多样性：混入text数据一起训（general-zh、en）
- 减少padding的长度
- whisper 是在30s的音频上做的训练，所以encoder只针对于30s的数据
- data efficiency 的训练



### Month Todo

- 长上下文探索：
  - Whisper
  - Mamba
  - Speech-mamba
  
- **Whisper 限制**

  

To Do：

- 整理一下代码

- a. 及时同步100h数据的实验结果；b.数据拉取；c.从100h抽6h做实验确认是否是数据质量问题

- 用中文的一些困难的ASR数据集去做训练和评测，带噪音的ASR数据

  ![image-20241212124630053](/Users/didi/Library/Application Support/typora-user-images/image-20241212124630053.png)

###  

Now 12.2

- bias 项目推进
  - maskgct搭建完成
- didi 项目推进
  - 裕hao
  - 和裕浩哥确认 base 模型, 他用的encoder是whisper，不适应于30s以上的模型（whisper限制了长上下文上）
  - 长上下文探索





10.30

- 微调：
- 评测代码整合
- 微调调研
- 长上下文调研







文本instruction更容易follow

音频instruction不是那么容易follow



10.29

- 微调方式调研：微调时长100h、多epoch、2 batch、数据长度、训练方式调研

- long context调研：whisper长上下文方向

- capstone

- bigdata paper

  

10.28

- 评测 
- 微调debug 
- capstone 

10.25

- 评测
- 微调框架debug



pytorch torchaudio torchvision 版本必须是对应的，如果不对应会导致无法运行

torchaudio torchvison新版本路径优于旧版本， 所以更换其他版本需要先卸载再重装

# Linux 常用指令

- 在 Shell 脚本中，`$((expression))` 是用于执行数学运算的正确语法，`$()` 内部是子命令替换的结构，而 `(( ))` 是数学运算的结构，

- `du -hs /path/to/dir` 主要用于查看**特定文件夹或文件**的大小，适合检查哪些文件或文件夹占用了大量空间。
  `df -h` 主要用于查看**文件系统或分区的剩余空间**，例如检查磁盘分区的容量是否即将用尽

- 软连接`ln -s <target> <link>` 

  <target>`要链接的文件或目录的实际路径，`<link>`创建软连接的路径或名称

  删除软连接 直接使用 rm删除 `<link>`

  `ls -l`验证软连接

  假设你有一个文件夹 `/home/user/documents`，你想在 `/home/user/Desktop` 下**创建**一个指向该文件夹的软连接

  ```python
  ln -s /home/user/documents /home/user/Desktop/documents_link
  ```

  

# Python Library

### 常用方法

- 当数据量很大时，生成器可以避免一次性加载整个数据集，节省内存。

  ```python
  def large_file_reader(file_path):
      with open(file_path, 'r') as file:
          for line in file:
              yield line
  
  for line in large_file_reader("large_file.txt"):
      print(line.strip())
  ```

- `data=open(path).readlines()`

- ```python
  from itertools import chain
  #chain 可以将多个可迭代对象连接起来，返回一个新的迭代器。
  #chain([1, 2, 3], [4, 5, 6]) 会将这两个列表连接成单一的序列 [1, 2, 3, 4, 5, 6]
  nested_list = [[1, 2, 3], [4, 5], [6]]
  flattened_list = list(chain(*nested_list))
  print(flattened_list)
  # 输出：[1, 2, 3, 4, 5, 6]
  ```

- `*` 用于位置参数解包（`*args`）和序列解包，将可迭代对象展开为单独的元素。

  - 使用 `*` 可以将多个列表解包为单一列表

    ```python
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    combined = [*list1, *list2]
    print(combined)
    # 输出：[1, 2, 3, 4, 5, 6]
    ```

  `**` 用于关键字参数解包（`**kwargs`）和字典解包，将字典展开为单独的键值对

  - 使用 `**` 可以将多个字典合并为一个新字典

    ```python
    dict1 = {"a": 1, "b": 2}
    dict2 = {"c": 3, "d": 4}
    combined = {**dict1, **dict2}
    print(combined)
    # 输出：{'a': 1, 'b': 2, 'c': 3, 'd': 4} 
    ```

- `partial` 是 Python `functools` 模块中的一个函数，用于**部分应用函数**。它的主要作用是**创建一个新函数，并为这个新函数预设一部分参数**，使得在调用时无需再指定这些参数

  ```python
  def chunk(sample, chunk_length=1024):
      # 处理分块逻辑
      pass
  # 等效于调用 chunk(sample, chunk_length=2048)
  new_chunk_function = partial(chunk, chunk_length=2048)
  new_chunk_function(sample)  # 自动使用 chunk_length=2048
  ```

- ```python
  line = "This is a test line"
  # 只按照第一个空格拆分
  parts = line.split(' ', maxsplit=1)
  print(parts)
  ```

  **`line.split(' ', maxsplit=1)`**：`split` 的 `maxsplit=1` 参数表示**最多拆分一次**，即只会根据第一个空格将 `line` 分成两部分。

  **返回结果**：返回一个包含两个元素的列表：

  - 第一个元素是第一个空格之前的内容
  - 第二个元素是第一个空格之后的全部内容（包括其他空格）

### 类变量 实例变量

| 特性         | 类变量（类属性）                       | 实例变量（实例属性）                      |
| ------------ | -------------------------------------- | ----------------------------------------- |
| **定义位置** | 类体内（通常在 `__init__` 方法前）     | `__init__` 方法中，使用 `self` 关键字定义 |
| **作用范围** | 整个类（所有实例共享）                 | 仅限于特定实例（每个实例独立）            |
| **内存使用** | 内存中只存储一份                       | 每个实例存储独立的变量，消耗更多内存      |
| **修改方式** | 通过类或实例修改，所有实例的值都会变化 | 只能通过实例修改，只影响该实例            |

类变量适用于所有实例共享的属性，实例变量适用于每个实例独立的属性。合理地使用类变量和实例变量有助于优化程序的内存和性能。



### 模块导入

模块就是后缀为`.py`的文件，包是一个包含模块的文件夹（或者是包含`__init__.py`文件的文件夹）

1. 导入语句应该位于文件的最上部
2. 导入应该根据导入的内容进行区分。通常有三种分类：

- 标准库导入(python内置库)
- 安装的第三方库(已经安装的并且不属于当前应用的模块)
- 本地应用导入(属于当前应用的模块)



#### 绝对导入

**绝对导入**指的是从项目的**根目录**开始，逐级指定模块路径进行导入。这种导入方式清晰直观，适合大型项目，尤其是在跨多个目录或包导入时。

`from package.subpackage.module import function_name` 

`import package.subpackage.module`

#### 相对导入

**当前工作目录**会根据您在终端运行脚本的位置动态变化。例如，如果在 `/home/user/project/` 目录下运行 Python 脚本，那么当前工作目录就是 `/home/user/project/`。

**相对路径**总是相对于**当前工作目录解释**的。

**导入模块**时，Python 会优先在**当前工作目录中查找模块**

**相对导入**是在当前模块或包的基础上，使用点号 (`.`) 表示**与当前模块的相对关系**来导入其他模块

- 一个点号代表了**相关的模块或者包**和当前的主位置在一个目录下

- 两个点号意味着在当前位置的父级目录下，也就是上一级目录

- 三个点号则意味着在上上层目录下，以此类推
- 当我们导入一个文件夹的时候，实际上我们是导入了该文件夹下面的`__init__.py`文件

```python 
from .module import function_name         # 当前目录
from ..subpackage import function_name    # 上级目录
from ...package import function_name      # 上上级目录
```

```python
my_project/
│
├── main.py
├── package/
│   ├── __init__.py
│   ├── module_a.py
│   └── subpackage/
│       ├── __init__.py
│       └── module_b.py
```

**示例代码**

如果我们在 `module_b.py` 中希望导入 `module_a.py`，可以使用相对导入：

```python
# module_b.py
from ..module_a import function_name
```

### 生成器

生成器是 Python 中一种特殊的迭代器，能够按需生成数据，节省内存并提高性能。生成器使用起来非常灵活，并且在需要大量数据但不希望一次性将所有数据加载到内存时非常有用

- 生成器的创建主要依赖于 `yield` 关键字。每次调用生成器时，它会返回一个新的值，并且会暂停，直到下一次被调用，继续从上次暂停的地方执行。

1. 生成器表达式

   生成器表达式类似于列表推导式，但使用 `()` 而不是 `[]`。生成器表达式会返回一个生成器对象，而不是立即生成整个列表

   ```python
   gen_expr = (x ** 2 for x in range(5))
   print(next(gen_expr))  # 输出: 0
   print(next(gen_expr))  # 输出: 1
   ```

2. 生成器函数

   生成器函数与普通函数类似，但使用 `yield` 而不是 `return` 返回值。**每次调用时，生成器函数会暂停在 `yield` 处**，并在下次调用时恢复执行

   ```python
   def countdown(n):
       while n > 0:
           yield n
           n -= 1
   
   gen = countdown(5)
   print(next(gen))  # 输出: 5
   print(next(gen))  # 输出: 4
   ```

   

### sys.argv

`sys.argv[1]` 是 Python 脚本中用于接收命令行参数的方式之一，它来自 `sys` 模块中的 `argv` 列表。为了更好地理解 `sys.argv[1]` 的作用，下面进行详细解释：

1. **`sys.argv` 概述**

- `sys.argv` 是 Python 中 `sys` 模块的一个属性，它是一个列表，包含传递给 Python 脚本的命令行参数。
- 列表的第一个元素，即 `sys.argv[0]`，总是表示**脚本的名称**（包括路径，取决于如何运行脚本）。
- 从 `sys.argv[1]` 开始，列表的后续元素代表命令行传入的其他参数。

2. **`sys.argv[1]` 详解**

- `sys.argv[1]` 代表传递给 Python 脚本的 **第一个命令行参数**（不包括脚本的名称）。
- 这是一个字符串，可以将其用于进一步的操作，比如文件路径、配置参数等。

示例：

假设有一个名为 `script.py` 的 Python 脚本，内容如下：

```python
import sys

print("Script name:", sys.argv[0])  # 输出脚本的名称
print("First argument:", sys.argv[1])  # 输出第一个命令行参数
```

如果我们从命令行运行此脚本，并传递参数：

```bash
python script.py input.txt
```

那么：

- `sys.argv[0]` 将是 `'script.py'`。
- `sys.argv[1]` 将是 `'input.txt'`，即传递给脚本的第一个参数。

输出结果为：

```
Script name: script.py
First argument: input.txt
```

3. **注意事项**

- **索引超出范围**：如果没有提供足够的命令行参数（如没有传递 `sys.argv[1]`），访问 `sys.argv[1]` 将会抛出 `IndexError`。例如：
  
  ```bash
  python script.py
  ```

  如果脚本尝试访问 `sys.argv[1]` 而没有传递参数，会出现以下错误：

  ```python
  IndexError: list index out of range
  ```

  **解决方法**：可以在代码中进行检查，确保参数的数量足够。例如：

  ```python
  import sys
  
  if len(sys.argv) > 1:
      print("First argument:", sys.argv[1])
  else:
      print("No argument provided.")
  ```

- **参数类型**：`sys.argv` 中的所有参数默认都是字符串类型。如果需要其他类型（如整数、浮点数），需要显式转换。例如：

  ```python
  number = int(sys.argv[1])  # 将第一个参数转换为整数
  ```

- **传递多个参数**：可以传递多个参数，`sys.argv[2]`、`sys.argv[3]` 等分别对应第二、第三个参数。例如：

  ```bash
  python script.py input.txt output.txt
  ```

  在此情况下：
  - `sys.argv[1]` 是 `'input.txt'`
  - `sys.argv[2]` 是 `'output.txt'`

4. **常见用途**

`sys.argv` 的常见用途包括：
- **文件路径**：用于传递输入、输出文件的路径。
- **命令行配置**：传递各种配置参数，例如模型的超参数、日志级别等。
- **脚本控制**：通过传递特定标记来控制脚本的行为。

总结

- `sys.argv[1]` 是从命令行接收的第一个参数（不包括脚本名）。
- 需要确保传递足够的参数，否则会引发 `IndexError`。
- 参数默认是字符串，必要时需要转换类型。



### round()

Python 中的 `round()` 函数用于将一个数字四舍五入到指定的小数位数。`round()` 可以处理浮点数和整数，它根据四舍五入规则调整数字的精度，并返回调整后的值。

 **`round()` 的基本语法**

```python
round(number, ndigits)
```

- `number`：要进行四舍五入的数字，可以是浮点数或整数。
- `ndigits`：（可选）表示要保留的小数位数。如果省略此参数，`round()` 会返回四舍五入到最接近的整数。

**`round()` 的工作原理**

`round()` 的核心是按照常见的“四舍五入”规则来处理数字：
- 如果数字的小数部分大于或等于 0.5，就进一位。
- 如果数字的小数部分小于 0.5，就舍去。

**示例 1：不指定 `ndigits` 参数**

```python
print(round(3.14159))   # 输出: 3
print(round(2.718))     # 输出: 3
print(round(1.499))     # 输出: 1
print(round(1.5))       # 输出: 2
```

在上面的示例中，`round()` 默认返回四舍五入后的**整数**。

**示例 2：指定 `ndigits` 参数**

```python
print(round(3.14159, 2))   # 输出: 3.14
print(round(2.71828, 3))   # 输出: 2.718
print(round(1.49999, 3))   # 输出: 1.5
```

在这个例子中，`ndigits` 参数指定了要保留的小数位数，`round()` 会根据四舍五入规则进行相应的舍入。

**特殊情况：`round()` 对于 .5 的处理**

Python 的 `round()` 在处理 `.5` 时采用的是**偶数舍入**（也叫**“银行家舍入”**）。这意味着当数字的小数部分正好是 `.5` 时，会将结果舍入到最接近的偶数，而不是总是向上舍入。

**示例 3：`.5` 的特殊情况**

```python
print(round(2.5))   # 输出: 2
print(round(3.5))   # 输出: 4
print(round(4.5))   # 输出: 4
print(round(5.5))   # 输出: 6
```

在这些例子中，`2.5` 被舍入为 `2`，而 `3.5` 被舍入为 `4`，这遵循银行家舍入法（四舍六入五成双）。

**`ndigits` 为负数的情况**

当 `ndigits` 为负数时，`round()` 会**将数字的小数点左侧的位数进行舍入**。例如，如果 `ndigits=-1`，那么数字会四舍五入到最接近的十位数。

```python
print(round(1234, -1))  # 输出: 1230
print(round(5678, -2))  # 输出: 5700
print(round(98765, -3)) # 输出: 99000
```

在这些例子中，数字会根据 `ndigits` 的值向左进行舍入。`round(1234, -1)` 表示四舍五入到最接近的十位数，而 `round(98765, -3)` 表示四舍五入到最接近的千位数。

**`round()` 和浮点数精度问题**

浮点数的精度问题是计算机在处理浮点数时的一个常见问题，`round()` 也可能受到这个问题的影响。在处理浮点数时，有时候可能会遇到意想不到的舍入结果，这是由于浮点数的二进制表示在某些情况下无法精确存储的原因。

```python
print(round(2.675, 2))  # 输出: 2.67 （不是 2.68）
```

在这个例子中，`2.675` 四舍五入到两位小数时，输出为 `2.67`，而不是预期的 `2.68`。这是因为 `2.675` 不能精确地表示为二进制浮点数，导致舍入结果出现了轻微误差。

 **`round()` 和 `format()` 的对比**

虽然 `round()` 可以四舍五入数字，但如果你需要控制数字的显示（比如保留固定的小数位），也可以使用 `format()` 或 f-string。

使用 `format()`：

```python
num = 3.14159
print("{:.2f}".format(num))  # 输出: 3.14
```

使用 f-string：

```python
num = 3.14159
print(f"{num:.2f}")  # 输出: 3.14
```

与 `round()` 不同，`format()` 和 f-string 更加关注**数字的显示格式**，而不改变数字本身的精度。





### BytesIO

`BytesIO` 是 Python 标准库 `io` 模块中的一个类，它提供了一个**在内存中处理字节流的方式，而不需要实际创建文件**。它常用于处理需要以字节流形式读写数据的场景，例如从网络下载数据、处理二进制文件等。

#### `BytesIO` 的基本用法

1. **导入模块**：
   
   ```python
   from io import BytesIO
   ```
   
2. **创建 `BytesIO` 对象**：
   ```python
   byte_stream = BytesIO()
   ```

3. **写入数据**：
   ```python
   byte_stream.write(b'Some binary data')
   ```

4. **读取数据**：
   ```python
   byte_stream.seek(0)  # Rewind the stream to the beginning
   data = byte_stream.read()
   print(data)  # Output: b'Some binary data'
   ```

#### 在 `librosa.load` 中使用 `BytesIO`

在你的代码示例中：

```python
from io import BytesIO
from urllib.request import urlopen
import librosa

audio_url = "http://example.com/audiofile.wav"
audio1, sr1 = librosa.load(
    BytesIO(urlopen(audio_url).read()),
    sr=processor.feature_extractor.sampling_rate
)
```

#### 代码详解

1. **从网络读取音频数据**：
   ```python
   urlopen(audio_url).read()
   ```
   这个调用会从指定的 `audio_url` 下载音频文件的内容，并将其读取为字节数据（`bytes`）。

2. **将字节数据包装为 `BytesIO` 对象**：
   ```python
   BytesIO(urlopen(audio_url).read())
   ```
   这里，`BytesIO` 将**下载的字节数据包装成一个文件-like 对象**，使得它可以像文件一样被处理。

3. **使用 `librosa.load` 读取音频数据**：
   
   ```python
   audio1, sr1 = librosa.load(
       BytesIO(urlopen(audio_url).read()),
       sr=processor.feature_extractor.sampling_rate
   )
   ```
   `librosa.load` 函数通常需要一个文件路径或一个类文件对象（如 `BytesIO`）作为输入。通过将 `BytesIO` 对象传递给 `librosa.load`，可以直接从内存中的字节流中读取音频数据，而不需要将其保存到磁盘上。`sr` 参数指定了要重采样到的采样率，`processor.feature_extractor.sampling_rate` 是目标采样率。

#### 总结

- **`BytesIO`** 提供了一种在**内存中读写字节数据的方式，可以用作文件-like 对象处理字节流**。
- 在这个例子中，它用于从网络读取音频数据并将其直接传递给 `librosa.load`，避免了将音频数据保存到磁盘的步骤。

这样可以更高效地处理从网络下载的音频数据，特别是在需要进行实时处理或不希望占用磁盘空间时。



### dict.from_keys()

```python
dict.fromkeys(seq[, value])
```

- **参数说明**：
  - `seq`：**必需**。表示字典的键集合，可以是任何可迭代对象，如列表、元组、字符串等。
    - 如果 `seq` 中有重复的元素，**字典的键会自动去重**，因为字典的键是唯一的
  - `value`：**可选**。设置为字典中所有键的默认值。如果未提供，默认为 `None`。
- **返回值**：返回一个新的字典，键来自于 `seq`，所有键的值都为 `value`

```python
list(dict.fromkeys(seq))  #去重后，将key转换为list
```



**基本用法**

**示例 1：创建值为 `None` 的字典**

```python
keys = ['name', 'age', 'gender']
new_dict = dict.fromkeys(keys)

print(new_dict)
# 输出：{'name': None, 'age': None, 'gender': None}
```

**示例 2：指定默认值**

```python
keys = ['apple', 'banana', 'cherry']
default_value = 0
fruit_dict = dict.fromkeys(keys, default_value)

print(fruit_dict)
# 输出：{'apple': 0, 'banana': 0, 'cherry': 0}
```

**示例：创建字典来统计字符出现次数**

```python
text = "hello world"
char_count = dict.fromkeys(text, 0)

print(char_count)
# 输出：{'h': 0, 'e': 0, 'l': 0, 'o': 0, ' ': 0, 'w': 0, 'r': 0, 'd': 0}
```

**示例：初始化配置参数**

```python
config_keys = ['host', 'port', 'username', 'password']
default_config = dict.fromkeys(config_keys, None)

print(default_config)
# 输出：{'host': None, 'port': None, 'username': None, 'password': None}
```

### 读写文件 with open

#### read

如果你使用 `for line in f` 逐行读取文件，换行符会包含在每行的字符串中：

```python
with open(input_path,'r') as f:
  for line in f :
     print(repr(line))  # 使用 repr() 打印，显示换行符
     #去掉换行符
     print(repr(line.strip()))
```

使用`readlines()` 读取文件的所有行

```python
with open('example.txt', 'r') as f:
    lines = f.readlines() # 返回一个列表，每行作为一个元素 # 每行末尾也会有换行符
    print(lines)  
```

如果你使用 `f.read()`，则会读取整个文件内容为一个字符串，换行符会保留在字符串中：

**`read(size=-1)`**：读取整个文件或指定字节数。`size` 参数控制读取字节数

```python
with open(input_path,'r') as f:
  content=f.read()
  print(repr(content))
```

**`readline(size=-1)`**：读取文件的一行。`size` 控制读取的字符数（而不是整行）。

```python
with open('example.txt','r',encoding='utf-8') as file:
  line1=file.readline() #first row
  line2=file.readline() #second row
```

#### write

**`write(string)`**：将字符串写入文件

```python
# write() 写入单行
with open('example.txt', 'w', encoding='utf-8') as file:
    file.write("Hello, World!\n")

    
fp=open('example.txt', 'w', encoding='utf-8')
fp.write(text)
```



### JSON

- **用途**：`json` 模块是 Python 标准库的一部分，用于处理 JSON 格式的数据。它能够将 Python 对象编码为 JSON 格式，或将 JSON 格式的数据解码为 Python 对象。

- **格式要求**：`json` 模块期望输入是有效的 JSON 格式，整个文件或字符串必须是一个合法的 JSON 文档。一般来说，JSON 文档的格式如下：
  - JSON 对象：`{"key": "value", "key2": "value2"}`
  - JSON 数组：`[{"key": "value"}, {"key": "value2"}]`

#### json.load

从**文件**读取JSON数据转换为python对象 （list, dict)

```python
import json

with open('data.json', 'r') as f:
    data = json.load(f)

```

#### json.loads

将JSON格式的**字符串**转换为python对象

```python
import json

json_str = '{"name": "Alice", "age": 25}'
data = json.loads(json_str)
print(data)#{'name': 'Alice', 'age': 25}
```

#### json.dump

将**python对象转换为JSON**格式**写入文件**

```python
import json

data = {"name": "Alice", "age": 25}

with open('output.json', 'w') as f:
    json.dump(data, f)

```

#### json.dumps

将**python对象**转换为**JSON格式字符串**

```python 
import json

data = {"name": "Alice", "age": 25}
json_str = json.dumps(data)
print(json_str)

```

##### case

`fp.write(json.dumps(result, ensure_ascii=False) + '\n')` 与 `fp.write(result + '\n')` 的区别在于是否将数据转换为 JSON 格式的字符串。

区别详解

1. **`json.dumps()`**：

   - `json.dumps(result, ensure_ascii=False)` 将 `result` 转换为 JSON 格式的字符串。如果 `result` 是一个字典、列表或其他复杂数据结构，`json.dumps()` 会将其转换为 JSON 字符串。
   - 使用 `ensure_ascii=False` 选项，确保非 ASCII 字符（如中文字符）不会被转义为 `\uXXXX` 形式。

   **示例**：

   ```
   python
   
   
   复制代码
   result = {"name": "张三", "age": 25}
   json_str = json.dumps(result, ensure_ascii=False)
   print(json_str)  # 输出: {"name": "张三", "age": 25}
   ```

2. **直接使用 `write()`**：

   - write() 只能填入字符串
   - 如果直接 `fp.write(result + '\n')`，假设 `result` 是一个字符串，会直接写入文件，不进行任何格式转换。
   - 但是，如果 `result` 是字典、列表或其他非字符串类型，直接 `write()` 会引发 `TypeError` 错误，因为 `write()` 只接受字符串。

   **示例**：

   ```
   python
   
   
   复制代码
   result = {"name": "张三", "age": 25}
   fp.write(result)  # 会报错：TypeError: write() argument must be str, not dict
   ```

何时使用 `json.dumps()`

- **复杂数据结构**（如字典、列表）：使用 `json.dumps()` 将数据转换为字符串，以便 `write()` 可以正常写入
- **需要保持 JSON 格式**：如果希望文件内容为 JSON 格式，以便之后可以直接解析，可以使用 `json.dumps()` 进行序列化

### jsonlines

`ensure_ascii=True`（默认）：非 ASCII 字符会被转义为 Unicode 序列（如 `\u4e2d\u6587`）

`ensure_ascii=False`：非 ASCII 字符会以其原始形式输出（如 `中文`）

- JSON Lines 格式要求每一行都是一个单独的、完整的 JSON 对象，行与行之间不需要逗号分隔，且不能包含额外的字符或数据。

  ```json
  {"name": "Alice", "age": 30}
  {"name": "Bob", "age": 25}
  ```

- **用途**：`jsonlines` 是一个第三方库，用于处理 JSON Lines 格式的数据。**JSON Lines 格式是一种文本格式，其中每一行都是一个独立的 JSON 对象**。这种格式特别适合处理大数据集，因为**可以逐行读取，而不需要一次性加载整个数据集**。

- **格式要求**：`jsonlines` 模块处理的是每行一个 JSON 对象的文件格式。例如：
  
  ```json
  {"key": "value1"}
  {"key": "value2"}
  ```
  
- **读取和写入**：
  - `jsonlines.Reader(file)`: 用于逐行读取 JSON Lines 格式的文件。
  - `jsonlines.Writer(file)`: 用于逐行写入 JSON Lines 格式的文件。

1. **写入 JSON Lines 文件**

使用 `jsonlines.Writer` 来创建一个新的 JSON Lines 文件并写入数据。

```python
import jsonlines

# 定义要写入的数据
data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
]

# 打开文件进行写入
with jsonlines.open('data.jsonl', mode='w') as writer:
    for entry in data:
        writer.write(entry)  # 每次写入一个 JSON 对象
```

在上面的代码中，`data.jsonl` 文件将包含以下内容：

```
{"name": "Alice", "age": 30}
{"name": "Bob", "age": 25}
{"name": "Charlie", "age": 35}
```

2. **读取 JSON Lines 文件**

使用 `jsonlines.Reader` 来逐行读取 JSON Lines 文件。

```python
import jsonlines

# 打开文件进行读取
with jsonlines.open('data.jsonl', mode='r') as reader:
    for obj in reader:
        print(obj)  # 每次读取一个 JSON 对象
```

输出将是：

```
{'name': 'Alice', 'age': 30}
{'name': 'Bob', 'age': 25}
{'name': 'Charlie', 'age': 35}
```

3. **批量写入和读取**

你还可以使用 `jsonlines` 提供的批量写入和读取功能，虽然这个库主要是逐行操作，但在处理大数据集时很有用。

批量写入

```python
import jsonlines

data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
]

with jsonlines.open('data.jsonl', mode='w') as writer:
    writer.write_all(data)  # 批量写入
```

批量读取

```python
import jsonlines

with jsonlines.open('data.jsonl', mode='r') as reader:
    data = reader.read_all()  # 批量读取
    print(data)
```

4. **错误处理**

`jsonlines` 模块在读取和写入时提供了错误处理机制。如果遇到无效的 JSON 行，会引发 `jsonlines.InvalidLineError`。

```python
import jsonlines

# 假设有一个无效的 JSON Lines 文件
try:
    with jsonlines.open('invalid_data.jsonl', mode='r') as reader:
        for obj in reader:
            print(obj)
except jsonlines.InvalidLineError as e:
    print(f"读取错误: {e}")
```

5. **使用上下文管理器**

`jsonlines` 支持使用上下文管理器（`with` 语句），确保文件在操作完成后正确关闭，这有助于避免资源泄露。



- **`jsonlines` 模块**：
  - 也要求每一行都是合法的 JSON 对象。如果某一行不是有效的 JSON，`jsonlines` 将会引发 `jsonlines.jsonlines.InvalidLineError`。
  - 不支持整体的非 JSON 格式的文件，因为 JSON Lines 格式要求每行都是独立的 JSON 对象。

- `json` 模块用于**处理整个 JSON 文档，期望输入是一个完整的合法 JSON 数据**。
- `jsonlines` 模块用于处理 JSON Lines 格式，期望输入是逐行的 JSON 对象。
- 两者都不支持非 JSON 格式的内容，它们都需要输入符合相应格式的有效 JSON 数据。如果输入格式不符合要求，将会抛出异常。



### ASCII 和 Unicode

Unicode编码和ASCII字符在计算机字符编码中有密切的关系，但它们有不同的范围和用途。以下是对两者关系和区别的详细说明：

**ASCII字符**

- **ASCII**（American Standard Code for Information Interchange，美国信息交换标准代码）是最早的字符编码标准之一，诞生于1960年代。
- **字符集范围**：ASCII 使用7位二进制编码，因此可以表示 **128** 个字符（2^7 = 128），范围是 **0 到 127**。这128个字符包括：
  - **控制字符**（0-31）：如回车（Carriage Return, CR）、换行（Line Feed, LF）等。
  - **可打印字符**（32-126）：包括数字（0-9）、英文字母（大小写 A-Z 和 a-z）、标点符号和一些特殊符号。
  - **删除字符**（127）：也称为删除控制符。

- **局限性**：ASCII 只能表示英文字母、数字和一些基本符号，不能表示其他语言中的字符，比如中文、日文、阿拉伯文等。

**Unicode编码**

- **Unicode** 是一种通用的字符编码标准，旨在为世界上所有语言中的所有字符分配一个唯一的代码点。它的设计目的是解决 ASCII 的局限性，支持不同语言、符号和特殊字符。
  
- **字符集范围**：Unicode 可以用不同的位数进行编码（8位、16位、32位），并能表示超过 **100万** 个字符，理论范围是从 **U+0000 到 U+10FFFF**。Unicode 为不同语言和符号提供了独立的编码空间，如中文、日文、阿拉伯文、表情符号等。
  
- **编码格式**：Unicode 定义了多个编码形式来表示字符，常见的有：
  - **UTF-8**：可变长度编码，使用1到4个字节编码字符。对于ASCII字符，它使用1个字节，与ASCII编码兼容。UTF-8 是互联网上最常用的编码方式，能够表示所有 Unicode 字符，同时保持与 ASCII 的兼容性。
  - **UTF-16**：使用2或4个字节编码字符。
  - **UTF-32**：使用固定的4个字节表示一个字符。

**关系**

- **向后兼容性**：Unicode 的设计使其与 ASCII 兼容。Unicode 的前128个代码点（**U+0000 到 U+007F**）完全与 ASCII 相同。因此，任何符合 ASCII 标准的字符在 Unicode 中保持不变。例如：
  - 字符 "A" 在 ASCII 中的编码是十进制的 65，二进制为 `01000001`，而在 Unicode 中的代码点也是 **U+0041**。
  
- **扩展性**：Unicode 扩展了 ASCII 的范围，不仅包括 ASCII 所涵盖的基本字符，还能编码世界上几乎所有的语言和特殊符号。Unicode 通过引入更多的字节数来表示更广泛的字符集，而不会影响到 ASCII 的使用。

**区别**

- **字符范围**：
  - **ASCII** 只能表示128个字符。
  - **Unicode** 能表示数百万个字符，涵盖全球多种语言和符号。
  
- **编码长度**：
  - **ASCII** 使用7位编码（在实际应用中通常扩展为8位，即1个字节）。
  - **Unicode** 可以使用1到4个字节来编码字符，具体取决于编码形式（如UTF-8、UTF-16、UTF-32）。

- **使用场景**：
  - **ASCII** 主要用于处理英文及其相关符号。
  - **Unicode** 用于支持多语言环境和国际化应用，可以处理世界上大部分语言字符。

**总结**

- **ASCII** 是字符编码的早期标准，适用于表示基本的英文字母和符号。
- **Unicode** 是一个现代化的字符编码标准，旨在支持全球各种语言和符号。**Unicode 的前128个字符与 ASCII 保持兼容**，从而使 Unicode 能够无缝支持和扩展 ASCII。



### shutil

`shutil` 是 Python 中的一个高级文件操作库，常用于文件和目录的复制、移动、删除等操作。与 `os` 模块中的一些基础文件操作相比，`shutil` 提供了更为强大和灵活的文件管理功能。接下来，我会详细讲解 `shutil` 的常用功能及用法。

`shutil` 模块的主要功能

1. **复制文件和目录** (`shutil.copy`, `shutil.copy2`, `shutil.copytree`)
2. **移动文件和目录** (`shutil.move`)
3. **删除文件和目录** (`shutil.rmtree`)
4. **压缩和解压缩文件** (`shutil.make_archive`, `shutil.unpack_archive`)
5. **磁盘使用情况** (`shutil.disk_usage`)
6. **文件/目录权限** (`shutil.chown`, `shutil.copymode`, `shutil.copystat`)

**1. 复制文件和目录**

**`shutil.copy(src, dst)`**

- 复制文件，从 `src` 到 `dst`。
- 如果 `dst` 是一个文件路径，文件会被复制到这个路径。
- 如果 `dst` 是一个目录路径，文件会被复制到该目录下，文件名不变。

```python
import shutil

shutil.copy('source_file.txt', 'destination_file.txt')  # 复制文件，保留权限但不保留元数据
shutil.copy('source_file.txt', '/path/to/directory/')  # 复制到目录
```

**`shutil.copy2(src, dst)`**

- 类似于 `shutil.copy`，但会复制文件的所有元数据（例如修改时间、权限等）。

```python
shutil.copy2('source_file.txt', 'destination_file.txt')  # 复制文件，同时复制元数据
```

**`shutil.copyfile(src, dst)`**

- 只复制文件的内容，不复制权限和元数据。
- 如果 `dst` 已存在，它将被覆盖。

```python
shutil.copyfile('source_file.txt', 'destination_file.txt')  # 仅复制内容
```

**`shutil.copytree(src, dst)`**

- 递归地复制整个目录树（包括所有子目录和文件）。
- `dst` 必须是一个不存在的路径，否则会报错。

```python
shutil.copytree('/path/to/source_dir', '/path/to/destination_dir')  # 递归复制目录
```

- 如果需要有选择地复制部分内容，可以使用 `ignore` 参数来跳过指定的文件或目录：
  
```python
def ignore_files(dir, files):
    return ['ignore_this_file.txt']  # 需要忽略的文件

shutil.copytree('/source_dir', '/destination_dir', ignore=ignore_files)
```

**2. 移动文件和目录**

**`shutil.move(src, dst)`**

- 移动文件或目录。
- 如果 `dst` 是目录，文件或目录会被移动到该目录下。
- 如果 `src` 和 `dst` 位于同一磁盘分区，它将直接重命名或移动文件。否则，它会先复制文件到新位置，然后删除旧文件。

```python
shutil.move('source_file.txt', 'destination_file.txt')  # 移动文件
shutil.move('/path/to/source_dir', '/path/to/destination_dir')  # 移动目录
```

**3. 删除文件和目录**

**`shutil.rmtree(path)`**

- 递归删除整个目录树。
- 这个操作会删除 `path` 指定的目录及其所有内容（包括文件和子目录），无法恢复。

```python
shutil.rmtree('/path/to/directory')  # 删除整个目录及其所有内容
```

- 你可以使用 `ignore_errors=True` 忽略删除过程中的错误。

```python
shutil.rmtree('/path/to/directory', ignore_errors=True)
```

**4. 压缩和解压缩文件**

**`shutil.make_archive(base_name, format, root_dir)`**

- 创建压缩文件（如 `.zip`, `.tar`）。
- `base_name` 是压缩文件的路径（不带后缀），`format` 是压缩格式（如 `'zip'`, `'tar'`），`root_dir` 是要压缩的文件或目录路径。

```python
shutil.make_archive('archive_name', 'zip', '/path/to/directory')  # 创建zip文件
```

**`shutil.unpack_archive(filename, extract_dir)`**

- 解压缩文件，`filename` 是压缩文件路径，`extract_dir` 是解压缩后的存储目录。

```python
shutil.unpack_archive('archive_name.zip', '/path/to/extract')  # 解压zip文件
```

**5. 磁盘使用情况**

**`shutil.disk_usage(path)`**

- 返回磁盘的使用情况，包括 `total`（总空间），`used`（已用空间）和 `free`（剩余空间）。

```python
import shutil

total, used, free = shutil.disk_usage('/')
print(f"Total: {total // (2**30)} GiB")
print(f"Used: {used // (2**30)} GiB")
print(f"Free: {free // (2**30)} GiB")
```

**6. 文件和目录权限**

**`shutil.chown(path, user=None, group=None)`**

- 更改文件或目录的所有者。
- `user` 和 `group` 可以是用户名或用户/组ID。

```python
shutil.chown('file.txt', user='username', group='groupname')
```

**`shutil.copymode(src, dst)`**

- 只复制文件的权限，不复制内容、所有权或其他元数据。

```python
shutil.copymode('source_file.txt', 'destination_file.txt')
```

**`shutil.copystat(src, dst)`**

- 复制文件的所有状态信息，包括权限、最后修改时间、所有权等。

```python
shutil.copystat('source_file.txt', 'destination_file.txt')
```

**7. 其他常用功能**

**`shutil.get_archive_formats()`**

- 返回支持的归档格式列表。

```python
print(shutil.get_archive_formats())  # [('bztar', 'bzip2 tar-file'), ('gztar', 'gzip tar-file'), ('zip', 'ZIP file')]
```

**`shutil.which(cmd)`**

- 返回可执行文件的路径，类似于 UNIX 的 `which` 命令。可以用于查找系统中某个命令的位置。

```python
shutil.which('python')  # 查找python的路径
```

**`shutil.get_terminal_size()`**

- 获取当前终端窗口的尺寸（行数和列数），通常用于打印动态输出。

```python
size = shutil.get_terminal_size()
print(f"Columns: {size.columns}, Rows: {size.lines}")
```

**`shutil` 和 `os` 的区别**

- `shutil` 提供了更高级的文件操作（如递归删除、目录复制、压缩等），而 `os` 提供了更底层的系统接口。
- 通常，使用 `shutil` 可以更方便地执行常见的文件管理任务，而 `os` 模块则适用于需要细粒度控制的场景。









## Huggingface

### Tokenizer

#### batch_decode

`batch_decode`批量解码token IDs

接收包含多个序列的列表，每个序列都是一组token IDs，然后将每个序列解码为对应字符串

```python
from transformers import AutoTokenizer

# 加载预训练的 tokenizer，例如 Bert
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 一组示例句子
sentences = ["Hello, how are you?", "I am fine, thank you!"]

# 将句子编码为 token IDs
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# 获取 token IDs
input_ids = encoded_inputs['input_ids']
print("Token IDs:")
print(input_ids)

# 将 token IDs 解码回文本
decoded_sentences = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
print("\nDecoded Sentences:")
print(decoded_sentences)
```

```css
Token IDs:
tensor([[  101,  7592,  1010,  2129,  2024,  2017,   102],
        [  101,  1045,  2572,  2986,  1010,  4067,  2017,   999,   102]])

Decoded Sentences:
['hello, how are you?', 'i am fine, thank you!']
```

##### arguments

`tokenizer.batch_decode(sequences, skip_special_tokens, clean_up_tokenization_spaces)`

**`sequences`**

- **类型**：`List[List[int]]` 或 `np.ndarray` 或 `torch.Tensor`
- **说明**：包含一批序列，每个序列是 token IDs 的列表。

**`skip_special_tokens`**

- **类型**：`bool`
- **默认值**：`False`
- **说明**：是否跳过特殊的 tokens（如 `[CLS]`、`[SEP]`、`[PAD]` 等）。

**示例：**

```python
# 不跳过特殊 tokens
decoded_with_special_tokens = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
print(decoded_with_special_tokens)
```

**输出：**

```css
['[CLS] hello, how are you? [SEP]', '[CLS] i am fine, thank you! [SEP]']
```

**`clean_up_tokenization_spaces`**

- **类型**：`bool`
- **默认值**：`True`
- **说明**：是否清理多余的空格。

**示例：**

```python
decoded_without_cleanup = tokenizer.batch_decode(input_ids, clean_up_tokenization_spaces=False)
print(decoded_without_cleanup)
```

**其他参数**

- **`spaces_between_special_tokens`**：在特殊 tokens 之间是否添加空格。



decode 循环处理单个token= batch decode 处理token序列

```python
decoded_sentences_loop=[tokenizer.decode(ids,skip_special_tokens=True) for ids in input_ids]

decode_sentencs_batch=tokenizer.batch_decode(input_ids,skip_special_tokens=True)
```



### Accelerate库

First, an empty model skleton is loaded into memory without using much RAM



分片读入模型权重

Next, a second model is loaded into memory with the weights of a single shard

Based on the passed in configuration, weights are stored in a variety of np.memmaps on disk or to a particular device

Then, the checkpoint is removed from memory through garbage collection

The offloaded weights are all sent to the CPU



Finally, hooks are added to each weight in the model to transfer the weights from CPU to GPU and back when needed

As the input reaches a layer, the hook triggers and weights are moved from the CPU to the GPU and back



量化对模型性能的影响

int8量化模型性能与float16格式差别不大

int4量化模型与float16模型相比，精度损失在1-2个百分点左右

## Pytorch

### torch.rsqrt

`torch.rsqrt()` 是 PyTorch 中的一个函数，用于计算输入张量中每个元素的 **平方根的倒数**（reciprocal square root）。它的数学表达式是：
$$
y = \frac{1}{\sqrt{x}}
$$
对于输入张量 `x`，该函数会返回一个新张量 `y`，其中每个元素对应于 `1 / sqrt(x_i)`，即输入张量每个元素平方根的倒数

1. **函数签名**

```python
torch.rsqrt(input, *, out=None) → Tensor
```

参数：

- **`input`**：输入的张量，类型是 `torch.Tensor`。
- **`out`**（可选）：接收结果的输出张量。如果提供，该张量的大小应与输入张量相同。计算结果将存储在 `out` 中。

返回值：

- 返回一个新的张量，每个元素是输入张量中相应元素的平方根倒数。

2. **示例**

下面是使用 `torch.rsqrt()` 的一些示例：

```python
import torch

# 创建一个张量
x = torch.tensor([4.0, 16.0, 25.0])

# 计算平方根的倒数
y = torch.rsqrt(x)
print(y)
```

**输出：**
```
tensor([0.5000, 0.2500, 0.2000])
```
- 解释：对于输入 `[4.0, 16.0, 25.0]`，输出是 `[1/sqrt(4), 1/sqrt(16), 1/sqrt(25)]`，即 `[0.5, 0.25, 0.2]`。

2.2 使用 `out` 参数：

```python
out_tensor = torch.empty(3)
torch.rsqrt(x, out=out_tensor)
print(out_tensor)
```

**输出：**
```
tensor([0.5000, 0.2500, 0.2000])
```
- 解释：这里结果直接存储在 `out_tensor` 中。

3. **计算特点**

3.1 负数输入

由于平方根运算仅适用于非负数，因此对于 **负数** 输入，`torch.rsqrt()` 会产生 `NaN`（Not a Number）或无效值，并会抛出警告。原因是负数的平方根在实数范围内没有定义。

示例：
```python
x = torch.tensor([-1.0, 4.0, 0.0])
y = torch.rsqrt(x)
print(y)
```

**输出：**
```
tensor([   nan, 0.5000,    inf])
```
- 解释：平方根的倒数对于负数结果是 `NaN`，对于 0 则是无穷大（`inf`），因为 \( 1/\sqrt{0} \) 是趋于无穷的。

4. **计算效率**

`torch.rsqrt()` 在深度学习中被广泛用于对张量中的元素进行高效地倒数平方根运算。它在处理**正则化、归一化、标准化**等操作时，常被用来优化性能。与直接计算 `1 / torch.sqrt(x)` 相比，`torch.rsqrt()` 可以实现更高效的计算。

例如，在实现批归一化（Batch Normalization）时，经常使用平方根倒数来计算标准差的倒数，进而进行归一化处理。

5. **与其他函数的对比**

- `torch.sqrt()`：计算每个元素的平方根。如果你只需要平方根而不是倒数，可以使用 `torch.sqrt()`：
  ```python
  x = torch.tensor([4.0, 16.0, 25.0])
  y = torch.sqrt(x)
  print(y)  # Output: tensor([2., 4., 5.])
  ```

- `1 / torch.sqrt(x)`：这个方法等价于 `torch.rsqrt(x)`，但是 `torch.rsqrt()` 更**高效且专为计算平方根的倒数而设计**，避免了额外的除法运算。

6. **应用场景**

- **Batch Normalization**：在 Batch Normalization 中，为了对输入数据进行标准化，我们需要除以标准差，通常会计算标准差的倒数（即平方根的倒数）。`torch.rsqrt()` 在这里能够提高计算效率。

- **权重正则化**：在某些深度学习算法中，平方根的倒数用于计算权重的调整或优化，尤其是使用 RMSprop 或 Adam 等优化器时。

- **归一化操作**：在处理信号或图像数据时，平方根倒数可能用于归一化和标准化操作。

7. **梯度计算**

`torch.rsqrt()` 也是一个可微函数，可以参与到 PyTorch 的自动微分机制中。在反向传播过程中，`torch.rsqrt()` 的导数会根据链式法则正确地计算。



### torch.cuda

在 PyTorch 中，管理 CUDA 设备（即 GPU）涉及多种方法，这些方法可以帮助用户查询设备信息、设置设备、以及管理设备间的数据传输等。这些功能主要集中在 `torch.cuda` 模块中。下面，我将详细解释一些常用的方法和它们的用途：

#### 1. 检查 CUDA 设备

- **`torch.cuda.is_available()`**:
  - 返回一个布尔值，指示 CUDA 是否可用于当前环境。
  - 这通常是判断是否可以在代码中使用 CUDA 功能的第一步。

- **`torch.cuda.device_count()`**:
  - 返回系统中可用的 CUDA 设备数量。
  - 这个函数对于在多 GPU 系统中进行分布式或并行计算非常有用。

#### 2. 设备上下文管理

- **`torch.cuda.set_device(device)`**:
  - 设置**当前 CUDA 设备**。
  - `device` 可以是设备的整数索引，例如 `0`，`1` 等，或者是设备的字符串名称，如 `'cuda:0'`。
  - 这个函数是用来指定所有后续 CUDA 操作应该在哪个设备上执行。

- **`torch.cuda.current_device()`**:
  - 返回当前选定的设备索引。
  - 有助于在多设备环境中检查当前设备的状态。

- **`torch.cuda.get_device_name(device=None)`**:
  - 获取 CUDA 设备的名称。
  - 如果没有指定设备，它将返回当前设备的名称。

#### 3. 数据传输和管理

- **`torch.cuda.memory_allocated(device=None)`**:
  - 返回指定 CUDA 设备上已分配的内存总量（单位为字节）。
  - 如果没有指定设备，它将返回当前设备上的内存使用情况。

- **`torch.cuda.memory_reserved(device=None)`**:
  - 返回为指定 CUDA 设备保留的内存总量。
  - 这通常大于或等于已分配的内存，因为 PyTorch 采用内存池技术来管理 CUDA 内存。

- **`torch.cuda.empty_cache()`**:
  - 释放 PyTorch 未使用的缓存内存，以便其他 GPU 应用程序可以利用这些内存。
  - 这不会影响已分配的 Tensor 内存，只是清除那些由 PyTorch 缓存管理器保留的内存。

#### 4. 设备间的数据移动

- **`tensor.to(device)`**:
  - 将 Tensor 移动到指定的设备上。
  - 例如，`tensor.to('cuda:0')` 会将 Tensor 移动到 GPU 0。
  - 这是将数据移动到 GPU 或从 GPU 移回 CPU 的标准方式。

#### 5. 错误处理和调试

- **`torch.cuda.set_device()`**:
  - 除了设置设备，这个函数在调试多 GPU 代码时也很有用，可以帮助确保操作被分派到正确的设备上。

- **`torch.cuda.synchronize(device=None)`**:
  - 等待特定设备上的所有核心完成当前的任务。
  - 这个函数对于调试和性能分析是非常重要的，因为它可以帮助确定操作的确切时间。

这些方法为在 PyTorch 中有效地使用 GPU 提供了必要的工具，使得 CUDA 设备的管理和操作更加简便和高效。在实际应用中，根据你的具体需求和设备的配置选择合适的方法非常关键。

### `torch.set_default_device()`

不受cuda_vision_devices影响

`torch.set_default_device()` 用于**设置默认设备**。设置后，所有没有明确指定设备的张量（tensor）都会在这个设备上创建。

**作用范围**：

- 影响全局的张量创建过程。设置了默认设备后，`torch.Tensor()` 和一些未指定设备的操作会使用这个设备。
- 可以用于任何设备类型，**不仅限于 CUDA（GPU），还可以是 CPU 或其他设备（如 MPS）**。

**示例**：

```python
import torch

# 设置默认设备为 CUDA:0
torch.set_default_device('cuda:0')

# 创建未指定设备的张量，默认在 CUDA:0 上创建
x = torch.tensor([1, 2, 3])
print(x.device)  # 输出: cuda:0
```

**注意**：

- 这影响的是张量的创建，而不是当前运行的 CUDA 设备上下文。

```python
torch.get_default_device()
#device(type='cpu')
torch.set_default_device('cuda')  # current device is 0
torch.get_default_device()
#device(type='cuda', index=0)

#torch.cuda.set_device()在这里修改了全局配置
torch.set_default_device('cuda')
torch.cuda.set_device('cuda:1')  # current device is 1
torch.get_default_device()
#device(type='cuda', index=1)
torch.set_default_device('cuda:1')
torch.get_default_device()
#device(type='cuda', index=1)
```



### `torch.cuda.set_device()`

`torch.cuda.set_device()` 用于**显式设置当前的 CUDA 设备上下**文。这个函数**只影响 GPU 设备**，并不影响 CPU 张量。通过这个函数，你可以在多 GPU 环境中指定当前的 GPU 设备，以便随后的 CUDA 操作和张量计算在该设备上进行。

**作用范围**：

- 仅**影响当前进程使用的 GPU 设备**。
- 后续在 CUDA 上运行的操作会使用这个设备，但是不影响张量的创建。如果没有明确指定张量的设备，默认会使用 `cuda:0`。

**示例**：

```python
import torch

# 设置当前的 CUDA 设备为 GPU 1
torch.cuda.set_device(1)

# 手动创建张量在当前设备上
x = torch.tensor([1, 2, 3], device='cuda')
print(x.device)  # 输出: cuda:1
```

**注意**：
- 它只对 GPU（CUDA 设备）有作用，对 CPU 张量无效。
- 仅设置当前的 CUDA 设备上下文，不改变张量的默认创建设备。

| **函数**                   | **适用范围**               | **作用**                                            | **影响范围**                                       |
| -------------------------- | -------------------------- | --------------------------------------------------- | -------------------------------------------------- |
| `torch.set_default_device` | 全局（适用于 CPU 和 CUDA） | 设置默认设备，未指定设备的张量会在该设备上创建      | 影响所有未指定设备的张量的创建                     |
| `torch.cuda.set_device`    | CUDA 设备                  | 设置当前的 CUDA 设备上下文，CUDA 操作在该设备上执行 | 仅影响 CUDA 操作的执行设备，不改变张量默认创建设备 |

## Numpy

NumPy 是 Python 中用于科学计算的核心库之一。它提供了高效的多维数组对象，以及大量的数学函数和操作，用于数组的创建、变换和计算。下面是一些 NumPy 常用的函数和方法的详细介绍。

### 1. **数组的创建**
NumPy 提供了多种创建数组的方式：

#### 1.1 `np.array()`
将 Python 列表或其他结构转换为 NumPy 数组。
```python
import numpy as np
a = np.array([1, 2, 3])
print(a)  # Output: [1 2 3]
```

#### 1.2 `np.zeros()`
创建一个指定形状的数组，所有元素初始化为 0。
```python
a = np.zeros((2, 3))  # 2 行 3 列的数组
print(a)
# Output:
# [[0. 0. 0.]
#  [0. 0. 0.]]
```

#### 1.3 `np.ones()`
创建一个指定形状的数组，所有元素初始化为 1。
```python
a = np.ones((3, 2))
print(a)
# Output:
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]]
```

#### 1.4 `np.empty()`
创建一个指定形状的空数组，但不初始化其值（数组中的元素是随机的）。
```python
a = np.empty((2, 2))
print(a)
```

#### 1.5 `np.arange()`
类似 Python 的 `range()` 函数，创建一个指定范围的数组，带步长。
```python
a = np.arange(0, 10, 2)
print(a)  # Output: [0 2 4 6 8]
```

#### 1.6 `np.linspace()`
在指定范围内返回指定数量的均匀分布的数。
```python
a = np.linspace(0, 1, 5)
print(a)  # Output: [0.   0.25 0.5  0.75 1. ]
```

### 2. **数组的形状操作**

#### 2.1 `reshape()`
改变数组的形状而不改变其数据。
```python
a = np.array([1, 2, 3, 4, 5, 6])
b = a.reshape((2, 3))  # 重塑为 2 行 3 列
print(b)
# Output:
# [[1 2 3]
#  [4 5 6]]
```

#### 2.2 `flatten()`
将多维数组转换为一维数组。
```python
a = np.array([[1, 2], [3, 4]])
b = a.flatten()
print(b)  # Output: [1 2 3 4]
```

#### 2.3 `T`（转置）
将数组的行与列互换（适用于多维数组）。
```python
a = np.array([[1, 2], [3, 4]])
b = a.T
print(b)
# Output:
# [[1 3]
#  [2 4]]
```

### 3. **数组的索引和切片**

#### 3.1 索引
NumPy 数组支持和 Python 列表类似的索引方式。
```python
a = np.array([10, 20, 30, 40])
print(a[2])  # Output: 30
```

#### 3.2 切片
对数组的子集进行选择。
```python
a = np.array([1, 2, 3, 4, 5])
print(a[1:4])  # Output: [2 3 4]
```

#### 3.3 布尔索引
使用条件筛选数组中的元素。
```python
a = np.array([1, 2, 3, 4, 5])
print(a[a > 3])  # Output: [4 5]
```

#### 3.4 花式索引
使用整数数组进行多次索引。
```python
a = np.array([1, 2, 3, 4, 5])
print(a[[0, 2, 4]])  # Output: [1 3 5]
```

### 4. **数组的数学操作**

#### 4.1 基本的算术操作
NumPy 支持数组的加减乘除等基本算术操作。
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 数组的逐元素相加、相乘
print(a + b)  # Output: [5 7 9]
print(a * b)  # Output: [ 4 10 18]
```

#### 4.2 广播（broadcasting）
当两个数组的形状不同但兼容时，NumPy 会自动扩展较小的数组来进行操作。
```python
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3])
print(a + b)
# Output:
# [[2 4 6]
#  [5 7 9]]
```

#### 4.3 聚合操作
- `sum()`：计算数组所有元素的和。
- `mean()`：计算数组的均值。
- `max()` / `min()`：找出数组中的最大值和最小值。

```python
a = np.array([1, 2, 3, 4, 5])
print(a.sum())  # Output: 15
print(a.mean())  # Output: 3.0
print(a.max())  # Output: 5
```

### 5. **随机数生成**

NumPy 提供了强大的随机数生成功能，常用方法如下：

#### 5.1 `np.random.rand()`
生成 [0, 1) 之间均匀分布的随机数。
```python
a = np.random.rand(3)
print(a)
```

#### 5.2 `np.random.randn()`
生成标准正态分布的随机数。
```python
a = np.random.randn(3)
print(a)
```

#### 5.3 `np.random.randint()`
生成指定范围内的随机整数。
```python
a = np.random.randint(1, 10, 3)  # 生成 3 个在 [1, 10) 之间的整数
print(a)
```

### 6. **线性代数**

NumPy 还提供了丰富的线性代数操作：

#### 6.1 `dot()`
计算两个数组的点积（矩阵乘法）。
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.dot(a, b))
# Output:
# [[19 22]
#  [43 50]]
```

#### 6.2 `transpose()`
矩阵转置。
```python
a = np.array([[1, 2], [3, 4]])
print(a.transpose())
# Output:
# [[1 3]
#  [2 4]]
```

#### 6.3 `linalg.inv()`
矩阵求逆。
```python
a = np.array([[1, 2], [3, 4]])
print(np.linalg.inv(a))
```

#### 6.4 `linalg.det()`
计算矩阵的行列式。
```python
a = np.array([[1, 2], [3, 4]])
print(np.linalg.det(a))
```

### 7. **数组的统计操作**

#### 7.1 `np.mean()`
计算数组的均值。
```python
a = np.array([1, 2, 3, 4])
print(np.mean(a))  # Output: 2.5
```

#### 7.2 `np.median()`
计算数组的中位数。
```python
a = np.array([1, 2, 3, 4])
print(np.median(a))  # Output: 2.5
```

#### 7.3 `np.var()` 和 `np.std()`
分别用于计算方差和标准差。
```python
a = np.array([1, 2, 3, 4])
print(np.var(a))  # 方差
print(np.std(a))  # 标准差
```

### 8. **排序和搜索**

#### 8.1 `np.sort()`
对数组进行排序。
```python
a = np.array([3, 1, 2])
print(np.sort(a))  # Output: [1 2 3]
```

#### 8.2 `np.argsort()`
返回数组排序后元素的索引。
```python
a = np.array([3, 1, 2])
print(np.argsort(a))  # Output: [1 2 0]
```

#### 8.3 `np.argmax()` 和 `np.argmin()`
返回数组中最大值和最小值的索引。
```python
a = np.array([1, 2
```

## Pandas

#### 选择数据

- **选择单列**：可以通过列名选择单个列。

  ```python
  print(df['Name'])  # 返回 'Name' 列的数据
  ```

- **选择多列**：选择多个列时，传递一个列名列表。

  ```python
  print(df[['Name', 'Age']])  # 返回 'Name' 和 'Age' 列的数据
  ```

- **选择行**：使用 `iloc[]` 选择基于整数位置的行，`loc[]` 选择基于标签的行

  ```python
  print(df.iloc[0])  # 选择第一行（基于整数位置）
  print(df.loc[0])  # 选择标签为 0 的行
  ```

- **切片**：使用 `iloc[]` 或 `loc[]` 对行和列进行切片

  ```python
  print(df.iloc[1:3])  # 选择第 2 到第 3 行
  print(df.loc[1:3, ['Name', 'City']])  # 选择第 2 到第 3 行的 'Name' 和 'City' 列
  ```

#### **条件筛选**

- **基于某列值的条件筛选**：

  ```python
  print(df[df['Age'] > 25])  # 筛选 'Age' 大于 25 的行
  ```

- **多个条件筛选**：使用 `&` (与) 和 `|` (或) 连接多个条件，并且每个条件都需要用括号括起来。

  ```python
  print(df[(df['Age'] > 25) & (df['City'] == 'NY')])  # 筛选 'Age' 大于 25 且 'City' 为 'NY' 的行
  ```

### 





#### 读取xlsx和csv文件

`pd.read_excel()` `pd.read_csv()`

- **XLSX**：

  - **文件类型**：`XLSX` 是 Excel 的电子表格文件格式，属于 Microsoft Office 的一种格式，是基于 XML 的文件格式（具体来说，是 `Office Open XML` 格式）。
  - **内容**：`XLSX` 文件包含多种数据类型（文本、数字、日期等），以及各种格式（字体、颜色、边框等），支持多个工作表、公式、图表、数据验证、条件格式、批注等复杂功能。
  - **结构**：`XLSX` 文件是一个压缩包，里面包含多个 XML 文件，用于存储不同类型的数据（例如，工作表内容、样式、公式等）。这些文件共同构成了 Excel 文件
  - `pandas` 使用 `openpyxl` 或 `xlrd` 引擎来处理 `XLSX` 文件，并通过它们解析 XML 格式的数据。
  - 在 `pandas` 中，读取 Excel 文件时有两个常用的引擎：
    - **`openpyxl`**：用于读取 `.xlsx` 文件（Excel 2007 及以上格式）。
    - **`xlrd`**：用于读取 `.xls` 文件（Excel 2003 及以下格式）。

- **CSV**：

  - **文件类型**：`CSV`（Comma-Separated Values）是一种简单的文本文件格式，每一行表示数据中的一条记录，字段之间通过逗号（`,`）分隔。`CSV` 文件通常用于存储表格数据，如电子表格中的数据。
  - **内容**：`CSV` 文件只是纯文本，通常只有列和行数据，没有格式、公式、图表等信息。它不支持数据类型（所有数据都是字符串），也不能存储格式化、公式等信息。
  - **结构**：`CSV` 文件是一种纯文本文件，数据按行分割，每行之间通常由换行符分隔，列之间由逗号或其他分隔符分隔

- 写入文件

  - ```python
    df.to_csv('output.csv', index=False)  # 保存为 CSV 文件
    df.to_excel('output.xlsx', index=False)  # 保存为 Excel 文件
    ```





# Basic Knowledge

### Logits

**Logits** 是指在神经网络中，经过最后一层线性变换（通常是一个全连接层）但还没有经过激活函数的输出值。它们是模型对每个类别的**原始预测分数**。在分类任务中，这些值通常会经过一个激活函数（如 softmax）来转换成概率分布，用于生成最终的分类预测。

Logits 的具体含义

1. **原始预测分数**: Logits 是模型在输出层的原始分数，这些分数通常没有被规范化为概率。它们表示模型对每个类别的信心，但不是最终的概率值。

2. **线性变换**: 在神经网络中，logits 通常是通过一个全连接层得到的，该层将前一层的激活值通过权重矩阵和偏置进行线性变换，输出一个向量。

3. **未激活的输出**: Logits 是未经过任何非线性激活函数（如 softmax）的输出值。它们可以是正数、负数或零，不受概率的约束。

**Logits 的数学表示**

假设你有一个分类任务，输出层的 logits 表示为 \( z = [z_1, z_2, ..., z_K] \)，其中 \( K \) 是类别的数量。这里的 \( z_i \) 是对第 \( i \) 类的原始预测分数。

**Logits 和 Softmax**

在多类别分类问题中，logits 通常会通过 softmax 函数转换为概率分布。Softmax 函数将 logits 转换为每个类别的概率值，公式如下：

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}} 
$$




其中 \($ z_i $\) 是第 \( i \) 类的 logits，\( $e$ \) 是自然对数的底数，分母是所有类别的 logits 的指数和。

Logits 和交叉熵损失

在分类任务中，交叉熵损失（cross-entropy loss）常用于评估模型的性能。交叉熵损失可以直接使用 logits 来计算。对于多类别分类，交叉熵损失的计算公式为：

$\text{Loss} = -\sum_{i} y_i \log(\text{softmax}(z_i))$ 

其中 \( $y_i$ \) 是实际类别的 one-hot 编码标签，\($\text{softmax}(z_i)$\) 是 logits 转换后的概率值。

Logits 的应用

1. **分类任务**: 在分类模型中，logits 是模型预测的核心，经过 softmax 函数得到的概率分布用于确定最终的类别预测。

2. **概率输出**: 通过对 logits 应用 softmax 函数，可以得到每个类别的概率值，用于模型的决策。

3. **损失计算**: 在训练过程中，使用 logits 直接计算交叉熵损失，可以提高数值稳定性，因为许多深度学习框架（如 TensorFlow 和 PyTorch）在实现中会处理 logit 和损失的计算，避免了对数计算带来的数值问题。

总结

- **Logits** 是神经网络中输出层的原始预测分数，经过线性变换但未经过激活函数。
- **Logits 转概率**: 通过 softmax 函数将 logits 转换为概率分布。
- **损失计算**: Logits 可直接用于计算交叉熵损失，通常比先转换为概率再计算更稳定。

### CNN

#### **常规卷积**

![image-20241016192205265](/Users/didi/Library/Application Support/typora-user-images/image-20241016192205265.png)

#### 深度可分离卷积

1）Depthwise 卷积

![image-20241016192401471](/Users/didi/Library/Application Support/typora-user-images/image-20241016192401471.png)

2）Pointwise 卷积

![image-20241016192634548](/Users/didi/Library/Application Support/typora-user-images/image-20241016192634548.png)

# LLM knowledege

### Autoregressive Model

自回归模型（Autoregressive Models）是一种用于时间序列或序列数据的模型，它通过依赖于自身的历史数据来生成当前或未来的预测。自回归模型的基本思想是，序列的下一个值是由前一部分数据（即先前的值）推导出来的

### Continual Pretraining

**Domain adaptation** is the process of customizing a generative AI foundation model (FM) that has been trained on massive amounts of public data to **increase its knowledge and capabilities for a specific domain or use case**. 

- This may mean adapting the model to excel at tasks in verticals like law, health, or finance, enhancing abilities in particular human languages, or personalizing the model to a company’s unique concepts and terminology. **Domain adaptation is a powerful technique for making generative AI models and solutions enterprise-ready.**

- Continued pre-training is often used for domain adaptation where the training set contains domain specific data such as manuals, documents, wiki pages, emails, new language, FAQs (Frequently Asked Questions). The pre-training process updates the model parameters and the model learns the domain knowledge, style, terminology and governing principles.

**Combination of domain specific and web data:** Providing only domain specific data can potentially degrade overall model performance for certain use cases. For example, if a model is adapted using vast amounts of text in a new language, it can forget logic learned when pre-trained on coding examples that it learned from English text during pre-training . As a result, a it is often advised to **augment the original dataset with non-domain specific high quality data**.

- **Quality-related pre-processing**, e.g. formatting, de-duplication (去重), PII filtering（Personally Identifiable Information Filtering，即个人身份信息过滤）.

- **NLP-related pre-processing**: 根据模型的上下文大小对输入数据进行分块

  ```python
  from itertools import chain
  from functools import partial
  from transformers import AutoTokenizer
  
  model_id = "meta-llama/Llama-2–13b-chat-hf" # alternatively "meta-llama/Llama-2–13b-hf"
  
  tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth_token=True) # previous authentication with HuggingFace needed, see here https://huggingface.co/docs/huggingface_hub/main/en/package_reference/login
  tokenizer.pad_token = tokenizer.eos_token
  
  # empty list to save remainder from batches to use in next batch
  remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
  
  def chunk(sample, chunk_length=2048):
    # define global remainder variable to save remainder from batches to use in next batch
    global remainder
    # Concatenate all texts and add remainder from previous batch
    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
    concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
    # get total number of tokens for batch
    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])
  
    # get max number of chunks for batch
    if batch_total_length >= chunk_length:
    batch_chunk_length = (batch_total_length // chunk_length) * chunk_length
  
    # Split by chunks of max_len.
    result = {
    k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
    for k, t in concatenated_examples.items()
    }
    # add remainder to global variable for next batch
    remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
    # prepare labels
    result["labels"] = result["input_ids"].copy()
    return result
  
  # tokenize and chunk dataset
  lm_dataset = dataset.map(
      lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
  ).map(
      partial(chunk, chunk_length=2048),
      batched=True,
  )
  
  # Print total number of samples
  print(f"Total number of samples: {len(lm_dataset)}")
  ```

  | 特性         | 预训练（Pre-training） | 持续预训练（Continual Pre-training） | 微调（Fine-tuning）                |
  | ------------ | ---------------------- | ------------------------------------ | ---------------------------------- |
  | **定义**     | 在通用大数据集上训练   | 在特定**领域数据**上进一步预训练     | 在**特定任务**标注数据集上训练     |
  | **数据类型** | 大规模通用数据集       | 领域特定未标注数据集                 | 任务特定标注数据集                 |
  | **目标**     | 获取通用知识           | 获取领域知识                         | 优化特定任务性能                   |
  | **应用场景** | 提供模型基础表示       | 领域适应（如医学、法律）             | 下游任务（如情感分析、问答、分类） |
  | **训练方式** | 自监督学习、无监督学习 | 无监督学习、自监督学习               | 有监督学习（标注数据）             |

### Emergent Abilities

在 Scaling Law 的框架下，随着规模的增加，模型性能提升

而 Emergent Ability 可能在这种性能提升的过程中出现，尤其是在规模达到某个阈值后，模型可能展现出全新的能力。

![image-20240916133238813](/Users/didi/Library/Application Support/typora-user-images/image-20240916133238813.png)

大模型的涌现能力现象：

- In context learning
- Chain-of-Thought

### Attention

在现代深度学习模型中，**Self-Attention**（自注意力）和 **Cross-Attention**（交叉注意力）是两个重要的注意力机制，尤其在 Transformer 架构中，它们在处理不同类型的任务中扮演关键角色。以下是对这两个注意力机制的详细解释、区别以及如何使用它们。

#### 1. Self-Attention（自注意力）

**Self-Attention** 是指在同一个输入序列中，**序列的每个位置可以关注该序列中的其他位置**。这种机制通过计算序列中各元素之间的相关性（即注意力权重），为每个元素生成上下文感知的表示。这在自然语言处理（NLP）任务中尤其重要，因为句子的某个单词可能依赖于其他位置的单词。

- **上下文依赖性**：通过注意力机制，输入序列中的每个位置可以根据其他位置的内容来更新自身表示。
- **并行计算**：不像循环神经网络（RNN）那样，Self-Attention 可以并行计算序列中的所有位置。
- **远距离依赖**：通过注意力机制，自注意力可以轻松捕捉远距离元素之间的依赖关系。

在 Transformer 编码器（Encoder）中，使用的就是 **Self-Attention** 机制，编码器处理的输入序列中的每个位置都可以关注同一个序列中的其他位置。

---

#### 2. Cross-Attention（交叉注意力）



**Cross-Attention** 是指**一个序列中的元素可以关注另一个序列中的元素**。这种机制在处理多模态任务或涉及不同数据源的任务中非常有用。典型的例子是在 Transformer 解码器（Decoder）中，解码器需要同时关注输入（编码器的输出）和解码器自身的输入。

#### 交叉注意力的特点：
- **跨序列依赖**：Cross-Attention 允许一个序列中的元素根据另一个序列的内容进行更新。
- **多模态任务**：在多模态任务中（如图像到文本的生成），文本可以对图像的表示进行关注，反之亦然。
- **解码器**：在 Transformer 解码器中，Cross-Attention 是用来将解码器的 Query 与编码器的输出（Key、Value）进行关联的关键机制。

在 Transformer 解码器（Decoder）中，Cross-Attention 机制用来将解码器的输入序列与编码器的输出序列关联起来。比如在机器翻译任务中，解码器通过 Cross-Attention 来参考编码器处理过的源语言表示，并生成目标语言输出。



#### 3. Self-Attention 和 Cross-Attention 的区别与联系

| 特性         | Self-Attention （自注意力）      | Cross-Attention （交叉注意力）                  |
| ------------ | -------------------------------- | ----------------------------------------------- |
| **输入来源** | 来自同一序列                     | 来自不同序列                                    |
| **计算对象** | Query、Key、Value 都来自同一序列 | Query 来自一个序列，Key 和 Value 来自另一个序列 |
| **应用场景** | 主要用于理解同一序列中的内部依赖 | 主要用于跨序列之间的信息交互                    |
| **使用位置** | Transformer 的编码器和解码器     | Transformer 的解码器、跨模态任务                |
| **示例**     | 机器翻译中的上下文捕捉           | 编码器-解码器架构中的源-目标关联                |

- **联系**：两者都是基于注意力机制的变体，核心的计算公式非常相似，都是通过 Query、Key、Value 的点积来计算注意力权重。
- **区别**：Self-Attention 是在同一序列中计算注意力，用于处理**序列中的内部依赖**；Cross-Attention 则是跨序列计算，**处理两个不同序列之间的信息交互**。

---



- **Self-Attention**: 自注意力机制用于同一序列中，每个元素对序列中其他元素进行关注，广泛用于 Transformer 的编码器和解码器中。
- **Cross-Attention**: 交叉注意力用于跨序列的注意力计算，一个序列的元素根据另一个序列的表示进行更新，常用于 Transformer 解码器以及多模态任务。

### Decoder only架构

几种主要架构：

- Encoder-only : BERT 用masked language modeling进行预训练，不擅长做生成任务，做NLU一般也需要有监督的下游数据微调
- encoder-decoder : T5 和 BART （T5最大参数量是11B，而11B是一个不借助PP，仅通过Zero+TP就可以训练的模型大小）
- Decoder-only : **GPT**、**LLama** (causal decoder) 用next token prediction做预训练，兼顾理解和生成，在各种下游任务上zero-shot和few-shot泛化性能都很好，
- Prefix LM: UNILM （non-causal decoder) (相对于GPT只改了attention mask，前缀部分是双向，后面要生成的部分是单向的causal mask)

**Decode-only 的泛化性能更好**：

- 用next token prediction预训练的decoder-only模型在各种下游任务上zero-shot泛化性能最好
- decoder-only 在few-shot（上下文学习）泛化能力更强（相比encoder-decoder 在in-context learning的学习上会更有优势，因为decoder-only的prompt可以更加直接地作用于每一层的参数）
- 上下文学习为decoder-only架构带来更好地few-shot性能： prompt和demonstration的信息可以视为对模型参数的隐式微调，decoder-only架构相比于encoder-decoder在in-context learning上会更有优势，因为prompt可以更加直接地作用于decoder每一层的参数，微调的信号更强

**注意力满秩问题**：

- 双向attention的注意力矩阵容易退化为低秩状态，而causal attention的注意矩阵是下三角矩阵，必然是满秩，建模能力更强

**预训练任务难度问题**：

- decoder-only模型**学习通用表征的上限**更高：bert的双向模型可以看到前向和后向，这在预测的时候是天然的优势，但在训练的时候其实降低了学习难度，换句话说：bert的双向提高了下限也拉低了上限，而当模型足够大，数据足够多的时候，decoder-only模型学习通用表征的上限更高

**隐式位置编码功能**：

- casual attention（decoder-only的单向attetention）具有**隐式的位置编码功能**，打破了transformer的位置不变性，而带有双向attention的模型，如果不带位置编码，双向attetention的部分token可以对换也不改变表示，对语序的区分能力天生较弱

**效率问题**：

- decoder-only一直服用**KV-cache**，对多轮对话更友好，因为每个token的表示只和它之前的输入有关，而encoder-decoder和Prifix LM就难以做到

路径依赖：

- 在工程生态上，decoder-only架构也形成了先发优势，比如flash-attention对causal attetion的支持就很好

**Infra优势**：

- decoder-only架构非常方便于scale up，基于scaling law的实际训练成本最低
- 流水并行（pipeline parallelism）是LLM分布式训练扩展到千卡集群以上的一个核心feature
- T5 的网络结构比 GPT 要复杂很多， T5 是 Encoder-Decoder 架构，整个网络分为两大块，且 Encoder 和 Decoder 的 Transformer Layer 参数大小、Attention 计算量、Context Length 等均不一致，导致 **Encoder 的理论计算量要比 Decoder 大很多**（**整个网络不是均匀对称的**）。 更要命的是， T5 Encoder 的输出要发给每个 Decoder Layer，网络结构不是线性而是有大量的分叉，前向反向之间包含了复杂的数据依赖关系， 会**导致流水并行中，各个 Stage 之间会产生大量的、非对称的、间隔跨多个 Stage 的数据依赖，**更加剧了流水并行的 load balance 问题。
- T5 Scale up 到 100B、500B 的难度很大，训练成本的增加远远高于 GPT。 因此也许 100B 的 T5 训练 10T tokens 的模型能力比 100B 的 GPT 更强，但为此要支付的算力/时间成本远大于 100B GPT 训练 10T token、

### Embedding

Embedding 将离散的输入(单词、句子、音频token等) 映射到连续向量空间的技术，在这个空间中每个输入都被表示为一个固定长度的向量

- 通过 embedding，模型可以将每个离散的输入（如单词 "dog" 或 "cat"）转换为一个高维向量（如 300 维的实数向量），使得这些**向量能捕捉到输入之间的语义关系**。

- Embedding 的核心思想是将**输入数据映射到一个新的表示空间**中，使得语义相似的输入（如 "dog" 和 "puppy"）在向量空间中彼此靠近

### LLama2

#### 大模型处理流程

1）输入数据

LLM的输入数据是一段文本，可以是一个句子或一段话。文本通常被表示成单词或字符的序列。

2）Tokenization

将输入数据切分成单词或字符，形成token序列

3）Embedding

将每个token映射为一个实数向量，为Embedding Vector

4）位置编码

对于token序列中的每个位置，添加位置编码（Positional Encoding）向量，以区分不同位置的token，为模型提供上下文关系的信息

(5)Transformer

在生成任务中，模型只需要用到Transformer 的decoder，即Decoder-Only，比如GPT、LLaMA 都是。

(6)自回归生成

在生成任务中，使用自回归（Autoregressive）方式，即**逐个生成输出序列中的每个Token**。
在解码过程中，每次生成一个Token时，**使用前面已生成的内容作为上下文，来帮助预测下一个Token。**

(7)输出处理

生成的Token序列通过一个输出层，通常是线性变换加上Softmax函数，将每个位置的概率分布转换为对应Token的概率

根据概率，选择概率最高的Token或者作为模型的预测结果



#### 模型结构

1. 前置的**RMSNorm**层
2. Q在与K相乘之前，先使用**RoPE**进行位置编码
3. **K V Cache**，并采用**Group Query Attention**
4. FeedForward层

##### RMSNorm

RMSNorm是LayerNorm的主要变体

与layerNorm相比，RMS Norm的主要区别在于**去掉了减去均值的部分**
$$
\begin{aligned} \text { RMSNorm : } y & =\frac{x}{\sqrt{\operatorname{Mean}\left(x^{2}\right)+\epsilon}} * \gamma, \ \operatorname{Mean}\left(x^{2}\right)  =\frac{1}{N} \sum_{i=1}^{N} x_{i}^{2}\end{aligned}
$$
```python
class RMSNorm(torch.nn.Module):
  def __init__(self,dim:int,eps:float=1e-6):
    super().__init__()
    self.eps=eps
    self.weight=nn.Parameter(torch.ones(dim))
  def _norm(self,x):
    return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
 	def forward(self,x):
    output=self.norm(x.float().type(x))
    return output*self.weight
```

给定输入向量 x，有 d 维：

1. **计算均方根（RMS）**：
   $$
   \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
   $$

2. **归一化**：
   $$
   \hat{x_i} = \frac{x_i}{\text{RMS}(x) + \epsilon}
   $$

3. **线性变换**：
   $$
   y_i = \gamma \hat{x_i}
   $$
   

   和 `LayerNorm` 不同，`RMSNorm` 只有一个缩放参数 $\gamma$，没有平移参数 $\beta$。

###### **LayerNorm BatchNorm**

`Layer Normalization` 是对一个层的神经元输出进行归一化，它是应用于 **每一个样本** 的所有特征（层的神经元），而不是批次的维度。这在小批次或序列数据中尤其有用，比如RNN等模型。

#### 公式
给定一个输入向量 \( x \)，它有 \( d \) 个维度（即特征），则 `LayerNorm` 的计算如下：

1. **计算均值**：
   $$
   \mu = \frac{1}{d} \sum_{i=1}^{d} x_i
   $$
   
2. **计算方差**：
   $$
   \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
   $$

3. **归一化**：
   $$
   \hat{x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
   $$
   
   
   其中， $\epsilon$  是一个很小的数，防止除以零。
   
4. **线性变换**：
   $y_i = \gamma \hat{x_i} + \beta$
   这里 \( $\gamma$ \) 和 \( $\beta$ \) 是可学习的缩放和平移参数，用于恢复模型的表达能力。

#### 优点
- **无依赖批次大小**：与`BatchNorm`不同，`LayerNorm`不依赖于批次大小，因此在小批次训练或递归神经网络中表现更好。
- **稳定性**：在各层中进行归一化，有助于稳定梯度的传播，避免梯度爆炸或消失。

#### 缺点
- **计算量大**：每次计算均值和方差时，涉及到所有特征，计算开销可能较大。

[Batch, Feature ]

**Batch Normalization**

- BN在每个特征维度上独立地进行归一化（竖着），但是归一化的统计量（均值和方差）是在一个小批量的数据上计算的；means that BN 依赖于批量的大小

- 在训练和推理阶段需要不同的行为：训练阶段，使用当前小批量统计量进行归一化；推理阶段，使用训练集上累积的统计量（移动平均of $\mu$ $\sigma$）进行归一化

- BN在处理图像数据（CNN）时特别有效，但在批量较小或处理序列数据（RNN）时会遇到问题

- ```python
  class BatchNorm1d:
  	def __init__(self,dim,eps=1e-5,momentum=0.1):
  		self.eps=eps
  		self.momentum=momentum
  		self.training=True
      self.gamma=torch.ones(dim)
      self.beta=torch.zeros(dim)
      #buffers
      self.running_mean=torch.zeros(dim)
      self.running_var=torch.ones(dim)
      
  	def __call__(self,x):
      if self.training:
      	xmean=x.mean(0,keepdim=True)
      	xvar=x.var(0,keepdim=True)
      else:
        xmean=self.running_mean
        xvar=self.running_var
      x_hat=(x-xmean)/torch.sqrt(xvar+self.eps)
      self.out=self.gamma*x_hat+self.beta
      if self.training:
        with torch.no_grad():
          self.running_mean=(1-self.momentum)*self.runing_mean+self.momentum*xmean
          self.running_var=(1-self.momentum)*self.running_var+self.momentum*xvar
     return self.out
  
  	def paramters(self):
      return [self.gamma,self.beta]
      
  ```

  

**Layer Normalizaiton**
$$
\begin{aligned} \text { LayerNorm }: y & =\frac{x-E[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta   ,\ E[x]=\frac{1}{N} \sum_{i=1}^{N} x_{i} ,\ \operatorname{Var}[x]  =\frac{1}{N} \sum_{i=1}^{N}\left(x_{i}-E[x]\right)^{2}\end{aligned}
$$


- LN在单个数据样本上进行归一化（横着），在所有特征维度上计算均值和方差；means that LN不依赖于小批量的大小，且在训练和推理阶段的行为是相同的
- LN适用于**小批次**或**批量大小变化较大**或**需要处理序列数据**的情况

```python
class LayerNorm1d:
  def __init__(self,dim,eps=1e-5):
    self.eps=dim
    self.gamma=torch.ones(dim)
    self.beta=torch.zeros(dim)
    
  def __call__(self,x):
    xmean=x.mean(1,keepdim=True)
    xvar=x.var(1,keedpdim=True)
    x_hat=(x-xmean)/torch.sqrt(xvar+self.eps)
    self.out=self.gamma*x_hat+self.beta
    return self.out
  
  def parameters(self):
    return [self.gamma,self.beta]
```



**指数移动平均** （Exponential Moving Average, EMA)

指数移动平均是一种对历史数据进行指数衰减加权的移动平均方法: MA 是通过递归公式来计算的，每个时间点的 EMA 是之前 EMA 和当前值的加权平均。



指数移动平均是一种对历史数据进行指数衰减加权的移动平均方法。EMA 相比于 SMA 和 WMA，能够更加迅速地对最新数据进行反应。与 WMA 不同，EMA 是通过递归公式来计算的，每个时间点的 EMA 是之前 EMA 和当前值的加权平均。

公式

EMA 的计算方式通过递归公式给出，给定平滑因子 $α$（0 <$ α$< 1），EMA 的递推公式为：
$$
EMA_t = \alpha \cdot x_t + (1 - \alpha) \cdot EMA_{t-1}
$$


其中 xtx_txt 是当前数据点，EMAt−1EMA_{t-1}EMAt−1 是前一个时刻的 EMA 值。

- **初始 EMA**：通常设定第一个 EMA 值为初始的简单移动平均值，即：

  $EMA1= \frac{1}{n} \sum_{i=1}^{n} x_i$

优点

- **快速响应最新变化**：EMA 对最近的数据点赋予了更高的权重，因此能够比 SMA 更快速地响应数据的变化。
- **更平滑的曲线**：由于历史数据的权重逐步衰减，EMA 能够在保留长时间趋势的同时，更好地平滑短期波动。

缺点

- **权重衰减不明显**：EMA 中较老的历史数据仍然会有一定影响，尽管其权重逐渐减小。
- **参数选择困难**：平滑因子 $\alpha$ 的选择对结果影响较大，通常需要根据数据特点进行调参

---

###### **RMSNorm（Root Mean Square Normalization）**

`RMSNorm` 是一种简化版的归一化方法，它不计算均值，而是通过直接计算特征的平方和的均值（均方根）来实现归一化。

#### 公式
给定输入向量 \( x \)，有 \( d \) 维：

1. **计算均方根（RMS）**：
   
   $\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}$
   
2. **归一化**：
   $\hat{x_i} = \frac{x_i}{\text{RMS}(x) + \epsilon}$
   
3. **线性变换**：
   $y_i = \gamma \hat{x_i}$
   
   和 `LayerNorm` 不同，`RMSNorm` 只有一个缩放参数 \( $\gamma$ \)，没有平移参数 \( $\beta$ \)。
   
   

优点

- **计算效率更高**：相比 `LayerNorm`，`RMSNorm` 不需要计算均值，方差和平方根操作相对较简单，因此更高效。
- **适合深度模型**：`RMSNorm` 在一些需要深度学习的场景（如大型预训练语言模型）中表现出色，因为它通过**更轻量的计算达到了类似的归一化效果**。

缺点

- **缺少平移操作**：由于没有平移参数 \($ \beta$ \)，`RMSNorm` 在某些情况下可能不如 `LayerNorm` 灵活。
- **不一定适合所有网络**：在某些网络结构中，去除均值和方差计算可能会影响模型性能。

---

### 总结

- **LayerNorm**：标准化输入的均值和方差，适合小批次数据或序列模型，具有可学习的缩放和偏移参数。
- **RMSNorm**：只对输入的均方根进行归一化，计算更高效，适合深层次的预训练模型，尤其是在效率要求高的场景。

如果你关心计算效率并且网络足够深，`RMSNorm` 可能是更好的选择；如果你更关心模型的稳定性和灵活性，`LayerNorm` 更加合适。

## SpeechLLM

### Basic

##### **时域 频域**

时域（时间域-time domain）——自变量是时间,即横轴是时间,纵轴是信号的变化。其动态信号x（t）是描述信号在不同时刻取值的函数

频域（频率域- frequency domain）——自变量是频率,即横轴是频率,纵轴是该频率信号的幅度,也就是通常说的频谱图。频谱图描述了信号的频率结构及频率与该频率信号幅度的关系。

对信号进行时域分析时，有时一些**信号的时域参数相同，但并不能说明信号就完全相同。因为信号不仅随时间变化，还与频率、相位**等信息有关，这就需要进一步分析信号的频率结构，并在频率域中对信号进行描述。动态信号从时间域变换到频率域主要通过傅立叶级数和傅立叶变换等来实现。

很简单时域分析的函数是参数是t，也就是y=f(t)，频域分析时，参数是w，也就是y=F(w)两者之间可以互相转化。时域函数通过傅立叶或者拉普拉斯变换就变成了频域函数

##### **梅尔刻度**

人耳对低频信号的区别更加敏感，而对高频信号的区别则不那么敏感：**频域上相等距离的两对频度，对于人耳来说他们的距离不一定相等**

梅尔频度-正常频度对应关系： **低频段的部分，梅尔刻度和正常频度几乎呈线性关系，而在高频段，因为人耳的感知变弱，因此两者呈对数关系**





![image-20241011192400299](/Users/didi/Library/Application Support/typora-user-images/image-20241011192400299.png)

##### **梅尔滤波器组**（mel-filter bank）

上面关系的矩阵表示就是 mel filter bank

mel filter bank的每一行对应一个梅尔频度，这一行的每个值表示每个频度对这个梅尔频度的权重

不同音频信号的mel-filter bank不同

##### **梅尔频谱**

梅尔频谱=mel filter bank * 功率谱

**功率谱** 是原始信号在频域上的能量表示

**梅尔频谱** 是功率谱经过mel滤波器组处理后的结果，强调了人耳对不同频率的感知

**从功率谱到梅尔频谱**

梅尔频谱的计算通常包括以下步骤：

- **计算功率谱**

  首先，对时间信号 x[n]进行短时傅里叶变换（STFT），得到频域信号 X[k]。然后计算其功率谱 P[k]：

  $P[k] = \frac{1}{N} |X[k]|^2$

- **应用mel滤波器组**

  将功率谱通过一组mel滤波器处理。mel滤波器组通常由若干个重叠的三角形滤波器组成，其输出通过以下公式计算：

  $P[k] \cdot H_{m}[k]M[m]=k$

降维: 原本数据的信息必然损失了一部分，但因为梅尔刻度是针对人耳设计的，因此梅尔频谱很大程度上保留了人耳理解原本语音所需的信息

##### 梅尔频率系数 MFCC

梅尔频率系数（Mel Frequency Cepstral Coefficients，MFCC）是一种广泛用于音频信号处理的特征表示，尤其在语音识别和音乐信息检索中。以下是MFCC的详细解释：

 **MFCC的基本概念**

MFCC是通过对音频信号进行处理而得到的一组特征，旨在模拟人耳对声音的感知特性。它们通过以下步骤计算而成。

2. **计算步骤**

2.1 **预加重**

对原始音频信号进行预加重，以强调高频部分：

$$
y[n] = x[n] - \alpha x[n-1]
$$

其中，\( $\alpha$ \) 通常取值在0.95左右。

2.2 **分帧**

将信号分成短时间帧，通常每帧长度为20到40毫秒，帧之间有重叠（例如50%重叠）。

2.3 **窗口化**

对每一帧应用窗口函数（如汉明窗），以减少边缘效应：

$$
w[n] = y[n] \cdot h[n]
$$
2.4 **快速傅里叶变换（FFT）**

对每一帧信号进行FFT，获得其频域表示。

2.5 **计算功率谱**

计算每帧的功率谱：

$$
P[k] = \frac{1}{N} |X[k]|^2
$$
2.6 **应用Mel滤波器组**

将功率谱通过一组梅尔滤波器进行处理，得到梅尔频率能量：

$$
M[m] = \sum_{k} P[k] \cdot H_{m}[k]
$$
2.7 **对数变换**

对每个梅尔频率能量取对数，以压缩动态范围：

$$
L[m] = \log(M[m])
$$
2.8 **离散余弦变换（DCT）**

对对数梅尔频率能量进行DCT，提取特征系数：
$$
C[n] = \sum_{m} L[m] \cos\left(\frac{\pi n}{M} \left(m + \frac{1}{2}\right)\right)
$$
其中，\( C[n] \) 是MFCC系数，\( M \) 是梅尔滤波器的数量。

3. **MFCC的特性**

- **低维表示**：MFCC通常提取12到13个系数，可以有效地压缩音频信号的信息。
- **频率响应**：通过梅尔滤波器组的非线性特性，更好地模拟人耳的听觉感知。
- **时间平稳性**：MFCC能够在一定程度上保持时间的稳定性，有利于后续的模式识别。

4. **应用**

MFCC被广泛应用于以下领域：

- **语音识别**：作为声学特征，用于训练和测试模型。
- **音乐分类**：用于识别和分类音乐风格或音色。
- **情感识别**：提取语音中的情感特征，帮助识别说话者的情绪状态。



### Whisper

弱监督预训练

弱监督68W小时数据 多样性音频数据 

为了证明大规模高质量有监督数据的有效性，论文直接使用**标准的transfomer结构（包含提取音频隐状态的encoder和进行文本标签自回归学习的decoder两个主体结构）**进行训练。计算方式包含几个主要的流程：首先，以25ms窗长、10ms步幅的参数提取fbank特征；然后，fbank特征过两个卷积层（为了降低特征复杂度第二层[卷积](https://zhida.zhihu.com/search?q=卷积&zhida_source=entity&is_preview=1)使用步幅为2进行2倍降采样）并加入[位置编码](https://zhida.zhihu.com/search?q=位置编码&zhida_source=entity&is_preview=1)；接着，再过标准的transformer-encoder进行[self-attention](https://zhida.zhihu.com/search?q=self-attention&zhida_source=entity&is_preview=1)计算得到音频的encoder-hidden_state；最后，**关键步骤是[decoder](https://zhida.zhihu.com/search?q=decoder&zhida_source=entity&is_preview=1)的自回归学习方式：根据当前标签预测下一个标签过程同时条件于encoder-hidden_state，具体实现方式是encoder-hidden_state（音频隐状态）和decoder需要预测的token（文本标注）之间计算两个序列的[cross-attention](https://zhida.zhihu.com/search?q=cross-attention&zhida_source=entity&is_preview=1)**，最终得到logits并和label输入到[交叉熵损失函数](https://zhida.zhihu.com/search?q=交叉熵损失函数&zhida_source=entity&is_preview=1)计算loss



卷积层对局部信息比较敏感

VAD（voice activity detection) 





CER（Character Error Rate）和 WER（Word Error Rate）都是用于评估文本识别系统（例如 OCR、语音识别系统）性能的指标。它们的主要区别在于计算的粒度和应用场景：CER 在**字符级别**衡量错误率，WER 在词级别衡量错误率。

**CER (Character Error Rate)**：

CER 是基于字符级别的错误率评估，计算模型预测输出与目标文本之间的字符错误率。其衡量的是单个字符的编辑距离（替换、删除、插入的字符数）与目标文本总字符数之间的比例。

公式：
$$
CER = \frac{S + D + I}{N}
$$

- \(S\)：替换（Substitution），字符被错误地识别为其他字符。
- \(D\)：删除（Deletion），某个字符在识别结果中缺失。
- \(I\)：插入（Insertion），多识别了错误的字符。
- \(N\)：目标文本的字符总数。

**WER (Word Error Rate)**：

WER 是基于词级别的错误率评估，计算模型预测输出与目标文本之间的词错误率。其衡量的是单个词的编辑距离（替换、删除、插入的词数）与目标文本总词数之间的比例。

公式：
$$
WER = \frac{S + D + I}{N}
$$

- \(S\)：替换（Substitution），一个词被错误地识别为其他词。

- \(D\)：删除（Deletion），目标文本中的某个词没有被识别出来。

- \(I\)：插入（Insertion），多识别了不属于目标文本的词。

- \(N\)：目标文本的词总数。

  

- **CER (字符级别)**：
  CER 是逐字符计算的，任何字符上的错误都会影响结果。它在细粒度的层次上检查预测输出的准确性。CER 适合用于对字符准确性要求高的任务，如 OCR、手写识别。

- **WER (词级别)**：
  WER 是逐词计算的，如果一个词整体被错误识别，计算时会将整个词视为错误。WER 适合用于语音识别等任务，评估系统是否正确识别出整个单词，忽略小的字符拼写错误。

3. **示例比较**：

假设目标文本是：
```
hello world
```
模型预测的输出是：
```
hxllo wrld
```

**计算 CER**：

- 替换：`x` 替代了 `e`，`r` 替代了 `o`。

- 总共发生了 2 个字符替换错误。

- 目标文本的字符总数为 10（包括空格）。

- 因此：
  $$
  CER = \frac{2}{10} = 0.2
  $$
  字符错误率为 20%。

**计算 WER**：

- 替换：`hxllo` 替代了 `hello`，`wrld` 替代了 `world`。
- 总共发生了 2 个词替换错误。
- 目标文本的词总数为 2。
- 因此：
  $$
  WER = \frac{2}{2} = 1.0
  词错误率为 100%。
  $$

**CER 适用场景**：

- **OCR（光学字符识别）**：例如将扫描文档中的字符提取为文本，细微字符错误可能导致误解，因此 CER 适用于评估字符级别精度。
- **手写识别**：评估手写字符识别的准确性时，CER 更为合适。
- **拼写检查**：用于评估拼写错误的系统。

**WER 适用场景**：

- **语音识别（ASR）**：在语音到文本转换中，词级别的错误更为重要。比如在语音识别任务中，识别出错误的单词比拼写错误影响更大，因此 WER 更能反映出系统在实际应用中的性能。
- **自然语言处理（NLP）**：WER 更适合评估机器翻译、摘要生成等任务，这些任务中完整单词或短语的正确性更为关键。

**CER 的优点和缺点**：

- **优点**：
  - 适合用于字符级别精确度要求高的任务。
  - 能捕捉微小的字符差异。
- **缺点**：
  - 对于一些应用，CER 可能过于严格。例如，如果一个字母错误导致整个词的错误，CER 可能会夸大错误。

**WER 的优点和缺点**：

- **优点**：
  - 在许多自然语言处理和语音识别任务中，词是语义的基本单位，WER 更贴近实际应用需求。
  - 小的拼写错误不会导致 WER 显著增加，容忍度更高。
- **缺点**：
  - 词级别的错误可能无法捕捉字符级别的细节差异。如果识别到的单词拼写略微有误但仍可以理解，WER 不会反映这种细微的错误。

**总结**：

- **CER** 适用于那些字符精度要求高的任务，如 OCR 和手写识别，它能捕捉到字符级别的微小错误。
- **WER** 更适合语音识别等任务，因为语义传递更依赖于词，而不是单个字符。

### Qwen2Audio

Problem

- 音频是否需要padding
- text item是否需要挤压
- 音频的读入是否不需要和text统一batch顺序
  - 我们使用dataloader按batch读入数据，为了保证数据按batch读入模型进行处理，text和audio在batch中要一一对齐

经过processor以后，得到 `text`: input_ids, attention_mask, `audio`: input_features, feature_attention_mask



Qwen2AudioForConditionalGeneration：Qwen2AudioConfig

- Qwen2AudioConfig(audio_config, text_config)：
  - audio_config : Qwen2AudioEncoderConfig
  - text_config:  Qwen2Config



Qwen2AudioEncoder： Qwen2AudioEncoderConfig

- input： input_features (generated from AutoFeatrureExtractor) **padding**



Dataset

给模型输入的数据 

`[`

​	`{text:text_content, audio: audio_path},`

​	`{text:text_content, audio: audio_path},`

`]`

- 分开 按batch和item对数据进行处理

- def collate_fn(self, batch) 按batch输出数据

- **`batch`**：在 `collate_fn` 方法中，`batch` 是一个列表，包含了批次中的所有样本。每个样本都是通过 `__getitem__` 方法获取的。

  **`collate_fn` 的作用**：将 `batch` 中的数据组合成适合模型输入的格式，通常需要处理序列长度不一致的问题，进行填充和转换为张量

# Paper Reading

#### Speechworthy Instruction-tuned Language Models

- Speech is serial and transient; speech processing is strictly linear  and requires **higher cognitive load** than reading

- Speech is serial and transient, and therefore **concise yet informative responses with conversational follow-up questions** are often preferred 
  通常首选简洁而信息丰富的回答以及对话式后续问题

  ```txt
  User: How should I choose a new phone?
  Assistant:
  There are manyfactors to consider whenchoosing a new phone,such as your budget, brandpreference and operating
  Would you like help narrowing down your options?
  ```

- removing prompts that require additional external context other than the request itself or those that explicitly ask for code or long form outputs that are unsuitable requests for speech-based interactions
  删除除请求本身之外还需要其他外部上下文的提示，或者明确要求代码或长格式输出的提示，这些提示不适合基于语音的交互请求。

- 



#### Recent Advances in Speech Language Models: A Survey

A Speech Language Model (SpeechLM) is an autoregressive foundation model that processes and generates speech data, utilizing contextual understanding for coherent sequence generation.

- Speech Language Models (SpeechLM) : end-to-end models , generate speech without converting from text
  - Encode speech waveforms into discrete tokens. they capture the **semantic information** of speech utterances and retain valuable **paralinguistic information,** which **prevents the information loss**
  - SpeechLMs effectively **mitigate the cumulative errors**, as their training is integrated with the speech encoding
  - SpeechLMs then model these tokens autoregressively, without solely relying on text input, which allows them to use the additional paralinguistic information to generate more expressive and nuanced speec. the generated tokens are synthesized back to speech

- A straightforward approach: ASR+LLM+TTS. suffering as **information loss** and **error accumulation**
- A semantic understanding speech **tokenizer** typically comprises a **speech encoder** and a **quantizer**, where the speech encoder encodes the essential information from the waveform and the quantizer discretizes continuous representations into discrete tokens



#### A Survey on In-context Learning

- **In-context learning**（上下文学习） is a paradigm that allows language models to learn tasks **given only a few examples** in the form of demonstration.  few-shot prompt属于ICL inference
  - **Prompt Learning**: prompts can be discrete templates or soft parameters that encourage the model to predict the desired output.
    - ICL can be regarded as a subclass of prompt tuning where the demonstration examples are part of the prompt
    - In-Context Learning 是 Prompt Learning 的一种特殊形式，但它的关键在于引入了示例数据
    - Prompt Learning 的主要目标是**通过适当的提示模板**，帮助模型在不调整模型参数的情况下，利用已有知识生成所需的输出。
  - **Few-shot Learning**（小样本学习）: few-shot learning is a general machine learning approach that involves adapting model parameters to perform a task with **a limited number of supervised examples**
    - ICL does not require parameter updates and is directly performed on pretrained LLMs.
    - Few-shot Learning 依赖于调整模型参数，使得模型可以适应少量的任务样本。这种方法适用于资源有限的场景，例如当数据稀缺时，希望模型仍然能够有效执行任务。
- ICL capabilities can be further enhanced through specialized training before inference
  - through **pretraining or continual pretraining**： reorganize pretraining corpora by aggregating related contexts, making models learn to reason across prior demonstrations.
  - finetune LLMs on a broad range of tasks with multiple demonstration examples, which boosts ICL abilities.

#### LIMA：Less is more for alignment

- Large language models are trained in two stages:
   (1) unsupervised pretraining from raw text, to learn general-purpose representations
  (2) large scale instruction tuning and reinforcement learning, to **better align to end tasks and user preferences**
- almost all knowledge in large language models is learned during **pretraining**, and only limited instruction tuning data is necessary to teach models to produce high quality output
- Ablation experiments reveal vastly diminishing returns when **scaling up data quantity** without also scaling up **prompt diversity**, alongside major gains when optimizing data quality.
- scaling up **input diversity** and **output quality** have measurable positive effects, while **scaling up quantity alone might not**.





# Linux

在 Linux 中，`$()` 和 `(( ))` 是两种常用的语法结构，分别用于 **命令替换** 和 **算术运算**。它们的功能和用法各不相同，但都在 Shell 编程中发挥重要作用。下面详细介绍它们的作用和使用方法。

**`$()` - 命令替换**

`$()` 是 **命令替换** 的语法，用于执行一个子命令并将其输出捕获为变量或作为另一个命令的参数。

**使用方法**

```bash
result=$(command)
```

这里，`command` 是任何可以在命令行中执行的命令，`$()` 会将命令的输出赋值给 `result` 变量。

**示例**

1. **捕获命令输出**

   ```bash
   date=$(date)
   echo "Today's date is: $date"
   ```

   **输出**：
   ```
   Today's date is: Mon Oct 30 12:34:56 UTC 2023
   ```

2. **作为其他命令的参数**

   ```bash
   echo "The current directory is $(pwd)"
   ```

   这里 `pwd` 命令的输出会被替换到 `echo` 语句中。

3. **嵌套命令替换**

   可以将 `$()` 嵌套使用，来执行多个命令。例如：

   ```bash
   files_count=$(ls $(pwd) | wc -l)
   echo "Number of files in the current directory: $files_count"
   ```

**2. `(( ))` - 算术运算**

`(( ))` 用于在 Shell 中执行整数算术运算，支持常见的加减乘除以及自增自减运算。它与 `$[]` 类似，但 `(( ))` 更常用，并且直接支持复杂表达式。

**使用方法**

```bash
(( expression ))
```

其中 `expression` 是要计算的算术表达式，`(( ))` 可以将表达式的值赋给变量，并且支持一些条件判断。

**示例**

1. **基本算术运算**

   ```bash
   a=5
   b=3
   (( sum = a + b ))
   echo "Sum: $sum"  # 输出: Sum: 8
   ```

2. **递增和递减**

   ```bash
   counter=0
   (( counter++ ))  # 自增，counter 变为 1
   echo "Counter: $counter"
   
   (( counter-- ))  # 自减，counter 变为 0
   echo "Counter: $counter"
   ```

3. **条件判断**

   `(( ))` 在条件语句中使用时，会根据表达式是否为非零来决定条件是否成立。

   ```bash
   a=10
   b=20
   if (( a < b )); then
       echo "$a is less than $b"
   else
       echo "$a is not less than $b"
   fi
   ```

4. **组合赋值与运算**

   可以将赋值和运算组合在一起：

   ```bash
   (( result = (a + b) * 2 ))
   ```

**3. 区别与联系**

| **语法** | **作用**           | **用法**                                     | **适用场景**                  |
| -------- | ------------------ | -------------------------------------------- | ----------------------------- |
| `$()`    | 命令替换           | `var=$(command)`                             | 获取命令输出，嵌套命令等      |
| `(( ))`  | 算术运算和条件判断 | `(( var = expression ))`, `(( expression ))` | 整数运算、自增/自减、条件判断 |

- **`$()` 执行的是 Shell 命令**，可以捕获命令输出；而 `(( ))` 是 **用于整数算术运算**，支持基本的数学计算和条件判断。
- **`$()` 可以嵌套其他命令**，而 `(( ))` 仅支持基本算术表达式。

**总结**

- **`$()`**：用于命令替换，将子命令输出捕获为变量或嵌入到其他命令中。
- **`(( ))`**：用于执行整数算术运算和条件判断，是 Shell 脚本中处理简单数学运算的常用方式。

这两种语法结构极大地方便了 Shell 编程，`$()` 用于灵活的命令处理，`(( ))` 用于高效的数值运算。

# GPU参考

| GPU                     | 显存  |
| ----------------------- | :---: |
| H200                    | 141GB |
| H100 H800               | 80GB  |
| A100 A800               | 80GB  |
| A100                    | 40GB  |
| V100                    | 32GB  |
| RTXA6000                |  48G  |
| RTX4090 RTX3090 A10 A30 | 24GB  |
| RTX4070                 | 12GB  |
| RTX3070                 |  8GB  |



Bias （speech in，text out）

目的 ：评测性别、人种的偏见

提供一个去偏的方法

统一英文

不同人群的知识偏见 MMLU测试





Goal： 测评SLM的性别和人种的偏见 并且提供一个去偏的方法

方法: 人种：黑人、白人、黄人
1. 搞一个 人种性别 平均的 语音代表人pool (现在是2x3种) 每个选3-5个 构成分布平均的语音pool

2. 测评方法 MASKGCT 翻译测评集

第一方面 知识测评集: MMLU抽10% 测评不同人群的知识偏见，acc比较

第二方面 偏见测评集: 代表例子就是 梦想职业和喜好什么的偏见 参考TTS现有LLM偏见测评集

Model：

- Qwen2Audio
- Gpt4o
- opensource SLM

3. 去偏方法： 

方法一: 
	无偏数据集的微调
	先用less is more 再用 Wizard
	tts问题，答案是文字 
	基于改平均人种的象征语音 TTS Wizard 或者 Less is more 给社区贡献一个分布平均的测评集

方法二: DO RLHF奖励无偏回答