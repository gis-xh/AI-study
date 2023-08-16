



https://github.com/lvwzhen/law-cn-ai



**Embedding**是一种将离散的符号（如单词、句子、段落或文档）表示为连续的向量或数组的方法，从而可以在高维空间中度量符号之间的语义和语法关系。Embedding是模型捕捉和存储语言知识的方式，也是模型编码和解码输入和输出文本的方式。Embedding可以帮助模型处理多模态任务，如图像和代码生成，也可以增强模型的文本分类、摘要、翻译和生成等能力。Embedding是基于Transformer架构的GPT系列模型的重要组成部分，它们可以根据模型和任务的不同而有不同的大小和维度。



**LLM**是**大型语言模型**（Large Language Model）的缩写，是一种利用海量文本数据训练出来的深度神经网络模型，可以处理各种自然语言处理任务，如文本生成、问答、对话、摘要等。LLM通常使用自回归（AR）或自编码（AE）的方式进行预训练，然后使用微调（Fine-tuning）或零样本（Zero-shot）的方式进行下游任务。LLM具有强大的泛化能力和知识存储能力，可以从文本中学习到丰富的语言规则、事实知识、常识知识等，并在适当的上下文中使用它们。LLM也具有一定的推理能力和创造能力，可以根据输入或提示生成新颖和合理的文本。LLM目前是人工智能领域最热门和最前沿的研究方向之一。



现在目标是根据本机实例搞清楚，大模型怎么用的，接口怎么调用，需要传哪些参数，以及调节哪些参数可以降低显存消耗

1、熟悉当前流行的各大模型的运行流程
2、找到尽可能配置小而全的大模型
3、熟悉使用现有的知识提取技术langchain
4、结合大模型与langchain实现基于用户专业领域知识的问答系统
……