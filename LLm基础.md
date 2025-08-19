# 1.1 N-gram模型
- N-gram 模型是NLP领域中一种基于统计的语言模型，广泛应用于语音识别、手写识别、拼写纠错、机器翻译和搜索任务。N-gram模型的核心思想是基于马尔可夫假设，即一个词的出现概率仅依赖于它前面的N-1个词
- 优点：实现简单、容易理解
- 缺点：N较大时，会出现数据稀疏性问题；模型忽略了词之间的范围依赖关系，无法捕捉到句子中的复杂结构和语义信息

# 1.2 Word2Vec
- Word2Vec是一种流行的词嵌入（Word Embedding）技术，基于神经网络NNLM的语言模型，旨在通过学习词与词之间的上下文关系来生成词的密集向量表示。
- Word2Vec模型主要有两种架构
    - 连续词袋模型（CBOW）：根据目标词上下文中的词对应的词向量，计算并输出目标词的向量表示；适用于小型数据集
    - Skip-Gram模型：利用目标词的向量表示计算上下文的词向量；适用于大型语料

# 1.3 Transformer
## 1.3.1 python实现注意力机制

```
def attention(query,key,value,dropout = None):
    # 获取K向量的维度，K与V的向量维度相同
    d_k = query.size(-1)
    # 计算Q与K的内积并除以根号d_K
    # transpose--相当于转置
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    #softmax
    p_atten = socres.softmax(dim=1)
    if dropout is not None:
        p_atten = dropout(p_atten)
    # 采样
#根据计算结果对V进行加权求和
return torch.matmul(p_atten,value),p_atten
```

## 1.3.2 多头注意力机制计算模块

```
import torch.nn as nn
import torch

'''多头自注意力计算模块'''
class MultiHeadAttention(nn.Module):

    def __init__(self, args: ModelArgs, is_causal=False):
        # 构造函数
        # args: 配置对象
        super().__init__()
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert args.dim % args.n_heads == 0
        # 模型并行处理大小，默认为1。
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行处理大小。
        self.n_local_heads = args.n_heads // model_parallel_size
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads

        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x n_embd
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，
        # 不理解的读者可以自行模拟一下，每一个线性层其实相当于n个参数矩阵的拼接
        self.wq = nn.Linear(args.dim, self.n_local_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_local_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_local_heads * self.head_dim, bias=False)
        # 输出权重矩阵，维度为 dim x n_embd（head_dim = n_embeds / n_heads）
        self.wo = nn.Linear(self.n_local_heads * self.head_dim, args.dim, bias=False)
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)
         
        # 创建一个上三角矩阵，用于遮蔽未来信息
        # 注意，因为是多头注意力，Mask 矩阵比之前我们定义的多一个维度
        if is_causal:
           mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
           mask = torch.triu(mask, diagonal=1)
           # 注册为模型的缓冲区
           self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = q.shape

        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, n_embed) -> (B, T, n_embed)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, C // n_head)，然后交换维度，变成 (B, n_head, T, C // n_head)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，
        # 然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        # 注意力计算
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # 计算 softmax，维度为 (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 做 Dropout
        scores = self.attn_dropout(scores)
        # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, C // n_head)，再拼接成 (B, T, n_head * C // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
```

## 1.3.3 前馈神经网络
```
class MLP(nn.Module):
    def __init__(self,dim:int,hidden_dim,dropout:float):
        super().__init__()
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim,hidden_dim,bias = False)
        # 定义第二层线性变换，从隐藏维度到输出维度
        self.w2 = nn.Linear(hidden_dim,dim,bias = False)
        # 定义Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        # 前向传播函数，输入x通过第一层线性变换和RELU激活函数
        # 然后通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.relu(self.w1(x))))
```
**注**：Transformer的前馈神经网络由两个线性层中间加一个RELU激活函数组成

## 1.3.4 层归一化（Layer Norm）
- 归一化的核心思想：将每层输入数据的分布强制拉回到一个标准正态分布（均值为，方差为1），从而稳定数据分布，加速训练，并允许使用更大的学习率。
- Batch Norm（批归一化）：
    - 对一个Batch内**同一特征通道**的所有激活值归一化，跨样本
- Layer Norm（层归一化）：
    - 对**单个样本**的所有激活值归一化，单一样本内
```
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
	super().__init__()
    # 线性矩阵做映射
	self.a_2 = nn.Parameter(torch.ones(features))
	self.b_2 = nn.Parameter(torch.zeros(features))
	self.eps = eps
	
    def forward(self, x):
	# 在统计每个样本所有维度的值，求均值和方差
	mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
	std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
    # 注意这里也在最后一个维度发生了广播
	return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

## 1.3.5 残差连接（下一层的输入=上一层的输入+上一层的输出）
```
# 注意力计算+残差网络
h = x + self.attention.forward(self.attention_norm(x))
# 经过前馈神经网络+残差网络
out = h + self.feed_forward.forward(self.fnn_norm(h))
```

## 1.3.6 Encoder
### Encoder 层
```
class EncoderLayer(nn.Module):
    def __init__(self,args):
        super().__init__()
        # 一层中有两个LayerNorm，分别再Attention之前和MLP之前
        self.attention_norm = LayerNorm(args.n_embd)
        # Encoder 不需要掩码，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x):
        # Layer Norm
        norm_x = self.attention_norm(x)
        # 自注意力
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out
```

### Encoder 块
```
class Encoder(nn.Module):
    '''Encoder 块'''
    def __init__(self, args):
        super(Encoder, self).__init__() 
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        "分别通过 N 层 Encoder Layer"
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

## 1.3.7 Decoder
### Decoder 层
```
class DecoderLayer(nn.Module):
  '''解码层'''
    def __init__(self, args):
        super().__init__()
        # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 MLP 之前
        self.attention_norm_1 = LayerNorm(args.n_embd)
        # Decoder 的第一个部分是 Mask Attention，传入 is_causal=True
        self.mask_attention = MultiHeadAttention(args, is_causal=True)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        # Decoder 的第二个部分是 类似于 Encoder 的 Attention，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        # 第三个部分是 MLP
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x, enc_out):
        # Layer Norm
        norm_x = self.attention_norm_1(x)
        # 掩码自注意力
        x = x + self.mask_attention.forward(norm_x, norm_x, norm_x)
        # 多头注意力
        norm_x = self.attention_norm_2(x)
        h = x + self.attention.forward(norm_x, enc_out, enc_out)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
```

### Decoder 块
```
class Decoder(nn.Module):
    '''解码器'''
    def __init__(self, args):
        super(Decoder, self).__init__() 
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)
```

## 1.3.8 搭建一个Transformer
- 流程：
    - 输入自然语言 ->通过分词器tokenizer将自然语言切分成token并转化为一个固定的index ->Embedding 层其实是一个存储固定大小的词典的嵌入向量查找表,将token转化为词向量表示 ->Encoding位置编码，加入到词向量编码中得到最终输入 -> Encoder块 -> Decoder块 -> 线性层 ->softmax层 ->最终输出
**Transformer 代码**
```
class Transformer(nn.Module):
    def __init__(self,args):
        super().__init__()
        #必须输入词表大小和block size
        asser args.vocab_size is not None
        asser args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size,args.n_embd),
            wpe = PositionalEncoding(args),
            drop = nn.Dropout(args.dropout),
            encoder = Encoder(args),
            decoder = Decoder(args),
        ))
        #最后的线性层，输入是n_embd，输出是词表大小
        self.lm_head = nn.Linear(args.n_embd,args.vocab_size,bias = False)

        # 初始化所有的权重
        self.apply(self._init_weights)

        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        # 统计所有参数的数量
        def get_num_params(self,non_embedding = False):
            # non_embedding: 是否统计 embedding 的参数
            n_params = sum(p.numel() for p in self.parameters())
            # 如果不统计 embedding 的参数，就减去
            if non_embedding:
                n_params -= self.transformer.wte.weight.numel()
        return n_params
        
        # 初始化权重
        def _init_weights(self,module):
            # 线性层和 Embedding 层初始化为正则分布
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # 前向计算函数
        def forward(self,idx,targets = None):
            #输入为idx，维度为（batch size,sequence length,1）:targets为目标序列，用于计算loss
            device = idx.device
            b,t = idx.size()
            assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"

        # 通过 self.transformer
        # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
        print("idx",idx.size())
        # 通过 Embedding 层
        tok_emb = self.transformer.wte(idx)
        print("tok_emb",tok_emb.size())
        # 然后通过位置编码
        pos_emb = self.transformer.wpe(tok_emb) 
        # 再进行 Dropout
        x = self.transformer.drop(pos_emb)
        # 然后通过 Encoder
        print("x after wpe:",x.size())
        enc_out = self.transformer.encoder(x)
        print("enc_out:",enc_out.size())
        # 再通过 Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:",x.size())

        if targets is not None:
            # 训练阶段，如果我们给了 targets，就计算 loss
            # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
            logits = self.lm_head(x)
            # 再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段，我们只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
```
## 2.1 GPT -- Decoder-only框架（主流LLM基础框架）
- 流程：一个自然语言文本的输入，先通过tokenizer进行分词转化为对应词典序号的input_ids. --> iput_ids通过Embedding层，再经过Positional Embedding 进行位置编码，变成hidden_states，然后进入解码器Decoder，由于不再有Encoder的编码输入，Decoder层仅保留了一个带掩码的注意力层，并且将LayerNorm层从注意力层后提到了注意力层之前。hidden_states进入Decoder层之后，先进行层归一化，再进行掩码注意力计算，然后经过残差连接和再一次层归一化进入到MLP中，结果为对应词表维度，然后转化为自然语言的token，从而生成我们的目标序列。

### 2.1.1 预训练任务--CLM（因果语言模型）
- 是N-gram语言模型的一个直接扩展。CLM是基于一个自然语言序列的前面所有token来预测下一个token。

### 2.1.2 ChatGPT
在GPT系列模型的基础上，通过引入**预训练--指令微调FT--人类反馈强化学习RLHF**的三阶段训练，OpenAI发布了跨时代的ChatGPT。

## 2.2 LLaMA -- Decoder Only
- LLaMA模型是由Meta（前Facebook）开发的一系列大型预训练语言模型。
- **过程**：与GPT类似。输入文本-->tokenizer进行编码 --> 变成input_ids --> 进入Embedding层映射到高维空间，转为词向量 --> 加上positional embedding层的编码，得到hidden_states --> 输入到（多个Decoder层） --> 首先进行RMSNorm，然后进入 masked 多头 self-attention 层 --> 残差网络 --> RMSNorm --> 前馈神经网络FNN（使用SwiGLU激活函数） --> 残差网络 --> 进入到下一个Decoder层-->...-->（输出层）线性层+softmax 将最终输出映射到词汇表，生成下一个token的概率分布

**相较于Transformer Decoder框架，Llama 3模型具体改进如下所示：**

- 使用**RMS Norm**代替了常用的Layer Norm，计算量减少 20% 且效果持平，加速训练；
- 激活函数由**SwiGLU**代替ReLU或是GELU，增强非线性表达能力。；
- 位置编码由原来的正弦-余弦绝对位置编码或是相对位置编码修改为**旋转位置编码RoPE**；
- 在70B模型中，采用 **GQA** 替代传统 MHA（Multi-Head Attention），将查询头分组共享键/值头，显著降低推理显存占用（约 30%），同时保持生成质量；在8B模型中，依然采用MHA结构。

## 2.3 GLM 智谱开发
- Decoder-Only架构
**与GPT的三点细微差异**：
- 使用**PostNorm**而非PreNorm。PostNorm--先进行残差连接，再进行LayerNorm；PreNorm--先进行LayerNorm计算，再进行注意力计算，最后残差计算。Pre Norm相对于因为有一部分参数直接加到了后面，不需要对这部分参数进行正则化，可以防止模型的梯度爆炸或梯度消失。
- 使用**单个线性层**实现最终token的预测，而不是使用MLP
- 激活函数从ReLU换成了**GeLUs**

# 3 LLM
## 3.1 训练LLM的三个阶段
**Pretrain预训练、SFT监督微调、RLHF人类反馈强化学习**

## 3.2 Pretrain
- 使用海量无监督文本对随机初始化的模型参数进行训练，它们的预训练任务也都沿承了 GPT 模型的经典预训练任务——**因果语言模型（Causal Language Model，CLM）**。

### 3.2.1 分布式训练框架
- 为了减少资源消耗，使用**分布式训练框架**。
- **核心思路**：数据并行和模型并行
- 目前主流的分布式训练框架包括Deepspeed、Megatron-LM、ColossalAI 等，其中**Deepspeed**使用最广。
### 3.2.2 Deepspeed
- 核心策略是 ZeRO 和 CPU-offload。
- ZeRO是一种显存优化的数据并行方案，核心思想是优化数据并行时每张卡的显存占用；

### 3.2.3 预训练数据的处理和清洗
- 预训练数据处理一般包括以下流程：
    1. 文档准备--包括URL过滤（根据网页URL过滤掉有害内容）、文档提取（从HTML中提取纯文本）、语言选择（确定提取的文本语种）等
    2. 语料过滤--核心目的是去除低质量、无意义、有毒有害的内容，如乱码、广告等。
        - 基于模型的方法：通过高质量语料库训练一个文本分类器进行过滤
        - 基于启发式的方法：通过人工定义web内容的质量指标，计算语料的指标值来进行过滤
    3.语料去重--大量重复文本会显著影响模型的泛化能力

## 3.3 SFT 有监督微调
- 预训练是LLM强大能力的根本来源。
- SFT的主要目标是让模型从多种类型、多种风格的指令中获得泛化的指令遵循能力，也就是能够理解并回复用户的指令。
- 模型的多轮对话能力完全来自于SFT阶段