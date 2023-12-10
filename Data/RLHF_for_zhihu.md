# RLHF

# RM

打分即标准。标准既包括回复的正确性，也包括回复的话术风格。

1. 要点覆盖全面
2. 要点回复正确
3. 语言流畅清晰，富有逻辑性
4. Follow既定话术、或者用户要求话术

先用高精0/1数据（有具体分值的数据）将模型打分分布拉开，然后使用pairwise对比数据训练。

细化打分维度，提升打分效果：从不同角度进行打分

### RM、PPO Query 的生成

## PPO

### 数据复用

### Policy Gradient

在给定状态s下(如给定query下)，我们希望优化policy来最大化期望得分：


<img src="https://www.zhihu.com/equation?tex=V_\theta(s)=\sum_\tau R(\tau|s)p_\theta(\tau|s)
" alt="V_\theta(s)=\sum_\tau R(\tau|s)p_\theta(\tau|s)
" class="ee_img tr_noresize" eeimg="1">

对所有状态求期望的整体优化目标是：


<img src="https://www.zhihu.com/equation?tex=O_\theta=\sum_s p_{\theta}(s)\sum_\tau R(\tau|s)p_\theta(\tau|s)=\sum_s p_{\theta}(s)V_\theta(s)
" alt="O_\theta=\sum_s p_{\theta}(s)\sum_\tau R(\tau|s)p_\theta(\tau|s)=\sum_s p_{\theta}(s)V_\theta(s)
" class="ee_img tr_noresize" eeimg="1">

实践过程中一般不对 <img src="https://www.zhihu.com/equation?tex=s" alt="s" class="ee_img tr_noresize" eeimg="1"> 的分布进行建模（未来可以考虑），而是直接从环境中采样，所以上式一般写成（下面的推导都是基于这一前提）：


<img src="https://www.zhihu.com/equation?tex=O_\theta=\sum_s p(s)\sum_\tau R(\tau|s)p_\theta(\tau|s)=\sum_s p(s)V_\theta(s)
" alt="O_\theta=\sum_s p(s)\sum_\tau R(\tau|s)p_\theta(\tau|s)=\sum_s p(s)V_\theta(s)
" class="ee_img tr_noresize" eeimg="1">

对 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 求导，有：


<img src="https://www.zhihu.com/equation?tex=\frac{\partial O_\theta}{\partial \theta} = \sum_s p(s) \sum_{\tau} p_{\theta}(\tau|s) R(\tau|s)\frac{\partial \log p_{\theta}(\tau|s)}{\partial \theta} = \sum_s p(s) \mathbb{E}_{\tau|s}[R(\tau|s)\frac{\partial log_\theta(\tau|s)}{\partial \theta}]
" alt="\frac{\partial O_\theta}{\partial \theta} = \sum_s p(s) \sum_{\tau} p_{\theta}(\tau|s) R(\tau|s)\frac{\partial \log p_{\theta}(\tau|s)}{\partial \theta} = \sum_s p(s) \mathbb{E}_{\tau|s}[R(\tau|s)\frac{\partial log_\theta(\tau|s)}{\partial \theta}]
" class="ee_img tr_noresize" eeimg="1">

一般采用随机梯度下降法，随机采样s以及从s开始的 <img src="https://www.zhihu.com/equation?tex=\tau" alt="\tau" class="ee_img tr_noresize" eeimg="1"> 来估算梯度，更新模型。

在模型拟合能力足够强的情况下，用这种方法训练出来的模型 <img src="https://www.zhihu.com/equation?tex=p_\theta" alt="p_\theta" class="ee_img tr_noresize" eeimg="1"> 最终会收敛到一个或者多个得分最高的 <img src="https://www.zhihu.com/equation?tex=\tau" alt="\tau" class="ee_img tr_noresize" eeimg="1"> ，其实非常类似于Reject sampling，或者SFT。其意义在于找到当前打分规则下最恰当的模型。当然，在有多个最高打分的情况下，模型的Diversity也会比只给一个输出下的SFT Diversity高。

在追求输出Diversity的情况下，Policy Gradient的训练轮次也不能太多。

MCTS 在exploration和explicition之间做折中，追求一定的Diversity。

### Q-Learning

Q-learning是同样是优化 <img src="https://www.zhihu.com/equation?tex=O_\theta" alt="O_\theta" class="ee_img tr_noresize" eeimg="1"> ，只不过采用的方法和Policy Gradient不同。令


<img src="https://www.zhihu.com/equation?tex=Q_\theta(s，a)=\sum_{s^\prime} p_\theta(s^\prime|s, a) \sum_\tau R(\tau|s^\prime)p_\theta(\tau|s^\prime)
" alt="Q_\theta(s，a)=\sum_{s^\prime} p_\theta(s^\prime|s, a) \sum_\tau R(\tau|s^\prime)p_\theta(\tau|s^\prime)
" class="ee_img tr_noresize" eeimg="1">

表示在s和policy  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 下采用动作a之后的期望得分。根据定义V、Q定义，期望奖励函数也可以写成如下形式：


<img src="https://www.zhihu.com/equation?tex=V_{\theta}(s) = \sum_{\tau}R(\tau)p_{\theta}(\tau|s)=\sum_{a}Q_{\theta}(s, a)p_{\theta}(a|s)
" alt="V_{\theta}(s) = \sum_{\tau}R(\tau)p_{\theta}(\tau|s)=\sum_{a}Q_{\theta}(s, a)p_{\theta}(a|s)
" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=\theta <img src="https://www.zhihu.com/equation?tex=的导数，而是把" alt="的导数，而是把" class="ee_img tr_noresize" eeimg="1"> Q_{\theta}(s, a) <img src="https://www.zhihu.com/equation?tex=当做一个常数，因此" alt="当做一个常数，因此" class="ee_img tr_noresize" eeimg="1"> O_{\theta}$最好写成：

" alt="\theta <img src="https://www.zhihu.com/equation?tex=的导数，而是把" alt="的导数，而是把" class="ee_img tr_noresize" eeimg="1"> Q_{\theta}(s, a) <img src="https://www.zhihu.com/equation?tex=当做一个常数，因此" alt="当做一个常数，因此" class="ee_img tr_noresize" eeimg="1"> O_{\theta}$最好写成：

" class="ee_img tr_noresize" eeimg="1">
O_{\theta} = \sum_s p(s)\sum_{a}Q(s, a)p_{\theta}(a|s)

<img src="https://www.zhihu.com/equation?tex=这里去掉 <img src="https://www.zhihu.com/equation?tex=Q_{\theta}" alt="Q_{\theta}" class="ee_img tr_noresize" eeimg="1"> 的下标 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 以表示在更新Policy Network参数的时候不会将Q视为 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 的函数，但Q仍然是在 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 状态下取得的值。

同时，根据Q，V的定义和LLM的特性（s_t, a_t到s_t+1的转移是确定的），又可以得到：

" alt="这里去掉 <img src="https://www.zhihu.com/equation?tex=Q_{\theta}" alt="Q_{\theta}" class="ee_img tr_noresize" eeimg="1"> 的下标 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 以表示在更新Policy Network参数的时候不会将Q视为 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 的函数，但Q仍然是在 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 状态下取得的值。

同时，根据Q，V的定义和LLM的特性（s_t, a_t到s_t+1的转移是确定的），又可以得到：

" class="ee_img tr_noresize" eeimg="1">
Q(s_t, a_t) = R(s_t, a_t, s_{t+1})+\gamma V(s_{t+1})

<img src="https://www.zhihu.com/equation?tex=所以可以学习一个Value Network 来学习 <img src="https://www.zhihu.com/equation?tex=V(s)" alt="V(s)" class="ee_img tr_noresize" eeimg="1"> 来计算 <img src="https://www.zhihu.com/equation?tex=Q(s, a)" alt="Q(s, a)" class="ee_img tr_noresize" eeimg="1"> 。在实际实践过程中，一般会计算优势函数来更新模型：

" alt="所以可以学习一个Value Network 来学习 <img src="https://www.zhihu.com/equation?tex=V(s)" alt="V(s)" class="ee_img tr_noresize" eeimg="1"> 来计算 <img src="https://www.zhihu.com/equation?tex=Q(s, a)" alt="Q(s, a)" class="ee_img tr_noresize" eeimg="1"> 。在实际实践过程中，一般会计算优势函数来更新模型：

" class="ee_img tr_noresize" eeimg="1">
A(s, a) = Q(s, a)-V(s) \\ O \leftarrow \sum_s p(s)\sum_a A(s, a) p_{\theta}(a|s) = \sum_s p(s)\sum_a (Q(s, a) - V(s)) p_{\theta}(a|s)

<img src="https://www.zhihu.com/equation?tex=之所以，可以用 <img src="https://www.zhihu.com/equation?tex=A(s, a)" alt="A(s, a)" class="ee_img tr_noresize" eeimg="1"> 替代 <img src="https://www.zhihu.com/equation?tex=Q(s, a)" alt="Q(s, a)" class="ee_img tr_noresize" eeimg="1"> 是因为：

" alt="之所以，可以用 <img src="https://www.zhihu.com/equation?tex=A(s, a)" alt="A(s, a)" class="ee_img tr_noresize" eeimg="1"> 替代 <img src="https://www.zhihu.com/equation?tex=Q(s, a)" alt="Q(s, a)" class="ee_img tr_noresize" eeimg="1"> 是因为：

" class="ee_img tr_noresize" eeimg="1">
\sum_s p(s)\sum_a - V(s) p_{\theta}(a|s) = -\sum_s p(s) V(s) \\ \frac{\partial \sum_s p(s) V(s) }{\partial \theta} = 0

<img src="https://www.zhihu.com/equation?tex=也即 <img src="https://www.zhihu.com/equation?tex=V(s)" alt="V(s)" class="ee_img tr_noresize" eeimg="1"> 的引入不会改变梯度的期望（随机梯度优化的目标），同时会降低梯度估计的方差[2](3.1.1)。计算梯度的方差对 <img src="https://www.zhihu.com/equation?tex=V(s)" alt="V(s)" class="ee_img tr_noresize" eeimg="1"> 的导数可知，当且仅当 <img src="https://www.zhihu.com/equation?tex=V(s)" alt="V(s)" class="ee_img tr_noresize" eeimg="1"> 等与 <img src="https://www.zhihu.com/equation?tex=Q(s, a)" alt="Q(s, a)" class="ee_img tr_noresize" eeimg="1"> 对 <img src="https://www.zhihu.com/equation?tex=a" alt="a" class="ee_img tr_noresize" eeimg="1"> 的期望时，梯度的方差最小，所以用上述的优势函数来更新模型。

**Value Network Learning**

一般Q，V是非常难以计算的，并且每次 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 更新之后Q，V也会随之改变，要每次准确计算Q，V在计算上是不可接受的。

Q-Learning使用**Bellman Equation来估计Q，V，进而优化 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 。同时在Deep RL框架下，会用一个神经网络（Value Network）来拟合V。**

![Untitled](https://raw.githubusercontent.com/v-mipeng/Markdown4Zhihu/master/Data/RLHF/Untitled.png)

![Untitled](https://raw.githubusercontent.com/v-mipeng/Markdown4Zhihu/master/Data/RLHF/Untitled 1.png)

Value network学习的核心思想是用采样得到的value值更新旧的value值，并让value network拟合新的value值。具体地又分为时间差分（TD）和蒙特卡洛采样（MC），二者的区别在于如何得到新值。

**时间差分TD算法**

 <img src="https://www.zhihu.com/equation?tex=Q(s_t, a_t) = R(s_t, a_t, s_{t+1})+\gamma V(s_{t+1})" alt="Q(s_t, a_t) = R(s_t, a_t, s_{t+1})+\gamma V(s_{t+1})" class="ee_img tr_noresize" eeimg="1">  是对 <img src="https://www.zhihu.com/equation?tex=V(s_t)" alt="V(s_t)" class="ee_img tr_noresize" eeimg="1"> 的一个估计，时间差分（TD）估计法就是依赖于此构建的：在 <img src="https://www.zhihu.com/equation?tex=s_t" alt="s_t" class="ee_img tr_noresize" eeimg="1"> 状态下采样一个 <img src="https://www.zhihu.com/equation?tex=a_t" alt="a_t" class="ee_img tr_noresize" eeimg="1"> ，用 <img src="https://www.zhihu.com/equation?tex=Q(s_t, a_t)" alt="Q(s_t, a_t)" class="ee_img tr_noresize" eeimg="1"> 来更新 <img src="https://www.zhihu.com/equation?tex=V(s_t)" alt="V(s_t)" class="ee_img tr_noresize" eeimg="1"> ：

" alt="也即 <img src="https://www.zhihu.com/equation?tex=V(s)" alt="V(s)" class="ee_img tr_noresize" eeimg="1"> 的引入不会改变梯度的期望（随机梯度优化的目标），同时会降低梯度估计的方差[2](3.1.1)。计算梯度的方差对 <img src="https://www.zhihu.com/equation?tex=V(s)" alt="V(s)" class="ee_img tr_noresize" eeimg="1"> 的导数可知，当且仅当 <img src="https://www.zhihu.com/equation?tex=V(s)" alt="V(s)" class="ee_img tr_noresize" eeimg="1"> 等与 <img src="https://www.zhihu.com/equation?tex=Q(s, a)" alt="Q(s, a)" class="ee_img tr_noresize" eeimg="1"> 对 <img src="https://www.zhihu.com/equation?tex=a" alt="a" class="ee_img tr_noresize" eeimg="1"> 的期望时，梯度的方差最小，所以用上述的优势函数来更新模型。

**Value Network Learning**

一般Q，V是非常难以计算的，并且每次 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 更新之后Q，V也会随之改变，要每次准确计算Q，V在计算上是不可接受的。

Q-Learning使用**Bellman Equation来估计Q，V，进而优化 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 。同时在Deep RL框架下，会用一个神经网络（Value Network）来拟合V。**

![Untitled](https://raw.githubusercontent.com/v-mipeng/Markdown4Zhihu/master/Data/RLHF/Untitled.png)

![Untitled](https://raw.githubusercontent.com/v-mipeng/Markdown4Zhihu/master/Data/RLHF/Untitled 1.png)

Value network学习的核心思想是用采样得到的value值更新旧的value值，并让value network拟合新的value值。具体地又分为时间差分（TD）和蒙特卡洛采样（MC），二者的区别在于如何得到新值。

**时间差分TD算法**

 <img src="https://www.zhihu.com/equation?tex=Q(s_t, a_t) = R(s_t, a_t, s_{t+1})+\gamma V(s_{t+1})" alt="Q(s_t, a_t) = R(s_t, a_t, s_{t+1})+\gamma V(s_{t+1})" class="ee_img tr_noresize" eeimg="1">  是对 <img src="https://www.zhihu.com/equation?tex=V(s_t)" alt="V(s_t)" class="ee_img tr_noresize" eeimg="1"> 的一个估计，时间差分（TD）估计法就是依赖于此构建的：在 <img src="https://www.zhihu.com/equation?tex=s_t" alt="s_t" class="ee_img tr_noresize" eeimg="1"> 状态下采样一个 <img src="https://www.zhihu.com/equation?tex=a_t" alt="a_t" class="ee_img tr_noresize" eeimg="1"> ，用 <img src="https://www.zhihu.com/equation?tex=Q(s_t, a_t)" alt="Q(s_t, a_t)" class="ee_img tr_noresize" eeimg="1"> 来更新 <img src="https://www.zhihu.com/equation?tex=V(s_t)" alt="V(s_t)" class="ee_img tr_noresize" eeimg="1"> ：

" class="ee_img tr_noresize" eeimg="1">
V_{new}(s_t) \leftarrow (1-\alpha) V_{old}(s_t) + \alpha Q_{old}(s_t, a_t, s_{t+1})=(1-\alpha) V_{old}(s_t) + \alpha \left(R(s_t, a_t, s_{t+1})+V_{old}(s_{t+1})\right)

<img src="https://www.zhihu.com/equation?tex=这里的 <img src="https://www.zhihu.com/equation?tex=\alpha" alt="\alpha" class="ee_img tr_noresize" eeimg="1"> 相当于随机梯度学习中的学习率。模型在学习过程中不断地用 <img src="https://www.zhihu.com/equation?tex=V_{old}" alt="V_{old}" class="ee_img tr_noresize" eeimg="1"> 生成 <img src="https://www.zhihu.com/equation?tex=V_{new}" alt="V_{new}" class="ee_img tr_noresize" eeimg="1"> ，然后更新Value network以拟合 <img src="https://www.zhihu.com/equation?tex=V_{new}" alt="V_{new}" class="ee_img tr_noresize" eeimg="1"> ，生成更好的价值评估模型。在得到 <img src="https://www.zhihu.com/equation?tex=V_{new}" alt="V_{new}" class="ee_img tr_noresize" eeimg="1"> 之后，用 <img src="https://www.zhihu.com/equation?tex=V_{new}" alt="V_{new}" class="ee_img tr_noresize" eeimg="1"> 得到新的 <img src="https://www.zhihu.com/equation?tex=O_{\theta}" alt="O_{\theta}" class="ee_img tr_noresize" eeimg="1"> 估计并优化Policy Network。

**蒙特卡洛MC**

此外,  <img src="https://www.zhihu.com/equation?tex=R(s_t, a_t)+\gamma R(s_{t+1}, a_{t+1}) + \cdots + R(s_{T-1}, a_{T-1}) + R(s_T,a_T)" alt="R(s_t, a_t)+\gamma R(s_{t+1}, a_{t+1}) + \cdots + R(s_{T-1}, a_{T-1}) + R(s_T,a_T)" class="ee_img tr_noresize" eeimg="1"> 也是对 <img src="https://www.zhihu.com/equation?tex=V(s_t)" alt="V(s_t)" class="ee_img tr_noresize" eeimg="1"> 的一次采样估计，蒙特卡洛算法（MC）就是据此构建的。在当前状态 <img src="https://www.zhihu.com/equation?tex=s_t" alt="s_t" class="ee_img tr_noresize" eeimg="1"> 下依概率生成后续的文本，然后利用该文本得到 <img src="https://www.zhihu.com/equation?tex=V(s_t)" alt="V(s_t)" class="ee_img tr_noresize" eeimg="1"> 的估计，用上面类似的方法对 <img src="https://www.zhihu.com/equation?tex=V(s_t)" alt="V(s_t)" class="ee_img tr_noresize" eeimg="1"> 进行更新:

" alt="这里的 <img src="https://www.zhihu.com/equation?tex=\alpha" alt="\alpha" class="ee_img tr_noresize" eeimg="1"> 相当于随机梯度学习中的学习率。模型在学习过程中不断地用 <img src="https://www.zhihu.com/equation?tex=V_{old}" alt="V_{old}" class="ee_img tr_noresize" eeimg="1"> 生成 <img src="https://www.zhihu.com/equation?tex=V_{new}" alt="V_{new}" class="ee_img tr_noresize" eeimg="1"> ，然后更新Value network以拟合 <img src="https://www.zhihu.com/equation?tex=V_{new}" alt="V_{new}" class="ee_img tr_noresize" eeimg="1"> ，生成更好的价值评估模型。在得到 <img src="https://www.zhihu.com/equation?tex=V_{new}" alt="V_{new}" class="ee_img tr_noresize" eeimg="1"> 之后，用 <img src="https://www.zhihu.com/equation?tex=V_{new}" alt="V_{new}" class="ee_img tr_noresize" eeimg="1"> 得到新的 <img src="https://www.zhihu.com/equation?tex=O_{\theta}" alt="O_{\theta}" class="ee_img tr_noresize" eeimg="1"> 估计并优化Policy Network。

**蒙特卡洛MC**

此外,  <img src="https://www.zhihu.com/equation?tex=R(s_t, a_t)+\gamma R(s_{t+1}, a_{t+1}) + \cdots + R(s_{T-1}, a_{T-1}) + R(s_T,a_T)" alt="R(s_t, a_t)+\gamma R(s_{t+1}, a_{t+1}) + \cdots + R(s_{T-1}, a_{T-1}) + R(s_T,a_T)" class="ee_img tr_noresize" eeimg="1"> 也是对 <img src="https://www.zhihu.com/equation?tex=V(s_t)" alt="V(s_t)" class="ee_img tr_noresize" eeimg="1"> 的一次采样估计，蒙特卡洛算法（MC）就是据此构建的。在当前状态 <img src="https://www.zhihu.com/equation?tex=s_t" alt="s_t" class="ee_img tr_noresize" eeimg="1"> 下依概率生成后续的文本，然后利用该文本得到 <img src="https://www.zhihu.com/equation?tex=V(s_t)" alt="V(s_t)" class="ee_img tr_noresize" eeimg="1"> 的估计，用上面类似的方法对 <img src="https://www.zhihu.com/equation?tex=V(s_t)" alt="V(s_t)" class="ee_img tr_noresize" eeimg="1"> 进行更新:

" class="ee_img tr_noresize" eeimg="1">
V_{new}(s_t) \leftarrow (1-\alpha) V_{old}(s_t) + \alpha ( R(s_t, a_t)+\gamma \left(R(s_{t+1}, a_{t+1}) + \cdots + R(s_{T-1}, a_{T-1}) + R(s_T,a_T)\right)

<img src="https://www.zhihu.com/equation?tex=![Untitled](https://raw.githubusercontent.com/v-mipeng/Markdown4Zhihu/master/Data/RLHF/Untitled 2.png)

**TD 和 MC的折中（GAE）**

  <img src="https://www.zhihu.com/equation?tex=R(s_t, a_t)+\gamma R(s_{t+1}, a_{t+1}) + \cdots + \gamma^i V(s_{t+i})" alt="R(s_t, a_t)+\gamma R(s_{t+1}, a_{t+1}) + \cdots + \gamma^i V(s_{t+i})" class="ee_img tr_noresize" eeimg="1"> 也是对 <img src="https://www.zhihu.com/equation?tex=V(s_t)" alt="V(s_t)" class="ee_img tr_noresize" eeimg="1"> 的一个采样估计，据此使用折中策略进行更新。

### 实践

在实际应用中，为了加速模型训练，我们可以约束Policy的概率分布，比如采用top-k，top-p的采样策略进行数据的采样。这有几方面好处：

1. 约束数据的采样空间，使其和RM数据的分布更契合，否则生成了RM无法正确评估的数据很容易导致评估失准，进而导致Value评估和Policy策略出问题。
2. 和生成过程契合，不会带来严重的负面影响，同时大大减小搜索空间，加速模型收敛。

在此基础上，在做Policy Network更新的时候，对于给定的query，可以考虑采样responses中， <img src="https://www.zhihu.com/equation?tex=V_{new}(s_T)" alt="V_{new}(s_T)" class="ee_img tr_noresize" eeimg="1"> 最大的response作为最佳Policy的近似，避免每一步的求最优值操作。这一操作的原因是在概率约束下，当前采样和最好的采样较为接近。

### Important Sampling

重采样的思想是利用已有的数据来估计当前Policy下的期望值。其核心原理是：

" alt="![Untitled](https://raw.githubusercontent.com/v-mipeng/Markdown4Zhihu/master/Data/RLHF/Untitled 2.png)

**TD 和 MC的折中（GAE）**

  <img src="https://www.zhihu.com/equation?tex=R(s_t, a_t)+\gamma R(s_{t+1}, a_{t+1}) + \cdots + \gamma^i V(s_{t+i})" alt="R(s_t, a_t)+\gamma R(s_{t+1}, a_{t+1}) + \cdots + \gamma^i V(s_{t+i})" class="ee_img tr_noresize" eeimg="1"> 也是对 <img src="https://www.zhihu.com/equation?tex=V(s_t)" alt="V(s_t)" class="ee_img tr_noresize" eeimg="1"> 的一个采样估计，据此使用折中策略进行更新。

### 实践

在实际应用中，为了加速模型训练，我们可以约束Policy的概率分布，比如采用top-k，top-p的采样策略进行数据的采样。这有几方面好处：

1. 约束数据的采样空间，使其和RM数据的分布更契合，否则生成了RM无法正确评估的数据很容易导致评估失准，进而导致Value评估和Policy策略出问题。
2. 和生成过程契合，不会带来严重的负面影响，同时大大减小搜索空间，加速模型收敛。

在此基础上，在做Policy Network更新的时候，对于给定的query，可以考虑采样responses中， <img src="https://www.zhihu.com/equation?tex=V_{new}(s_T)" alt="V_{new}(s_T)" class="ee_img tr_noresize" eeimg="1"> 最大的response作为最佳Policy的近似，避免每一步的求最优值操作。这一操作的原因是在概率约束下，当前采样和最好的采样较为接近。

### Important Sampling

重采样的思想是利用已有的数据来估计当前Policy下的期望值。其核心原理是：

" class="ee_img tr_noresize" eeimg="1">
\mathbb{E}_p[x]=\mathbb{E}_q[x\frac{p(x)}{q(x)}]

<img src="https://www.zhihu.com/equation?tex=在Policy Gradient中，我们用采样的方法估计模型梯度：

" alt="在Policy Gradient中，我们用采样的方法估计模型梯度：

" class="ee_img tr_noresize" eeimg="1">
\frac{\partial O_\theta}{\partial \theta} \leftarrow \frac{1}{nm}\sum_{i=1}^n \sum_{j=1}^m R(\tau_j|s_i)\frac{\partial \log p_\theta(\tau_j|s_i)}{\partial \theta}, (s_i, \tau_j) \sim p_{\theta}(s, \tau)

<img src="https://www.zhihu.com/equation?tex=这里， <img src="https://www.zhihu.com/equation?tex=(s_i, \tau_j)" alt="(s_i, \tau_j)" class="ee_img tr_noresize" eeimg="1"> 都是在Policy为 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 时的采样。当我们想用之前从policy为 <img src="https://www.zhihu.com/equation?tex=\theta^\prime" alt="\theta^\prime" class="ee_img tr_noresize" eeimg="1"> 时采样的样本估计当前模型的梯度时，可以改写上式：

" alt="这里， <img src="https://www.zhihu.com/equation?tex=(s_i, \tau_j)" alt="(s_i, \tau_j)" class="ee_img tr_noresize" eeimg="1"> 都是在Policy为 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 时的采样。当我们想用之前从policy为 <img src="https://www.zhihu.com/equation?tex=\theta^\prime" alt="\theta^\prime" class="ee_img tr_noresize" eeimg="1"> 时采样的样本估计当前模型的梯度时，可以改写上式：

" class="ee_img tr_noresize" eeimg="1">
\frac{\partial O_\theta}{\partial \theta} \leftarrow \frac{1}{nm}\sum_{i=1}^n \sum_{j=1}^m \frac{p_{\theta}(s_i, \tau_j)}{p_{\theta^\prime}(s_i, \tau_j)} R(\tau_j|s_i)\frac{\partial \log p_\theta(\tau_j|s_i)}{\partial \theta}, (s_i, \tau_j) \sim p_{\theta^\prime}(s, \tau)
$$

通过这种方法可以达到利用已有采样样本的目的，提高样本利用率。同样的思想可以运用在Q-Learning中。

方差：

### Tricks

不同样本之间的竞争与促进？

1. Advantage normalization across samples
2. Reward Clipping （不减去均值，只除以标准差）

这两者都是在做Cross-batch的尺度归一

### Questions

- 怎么利用相对值进行学习？
    - Reject Sampling (可以不用非绝对打分)
- 怎么用多RM进行学习？

# Reference

1. [https://hackmd.io/@shaoeChen/Bywb8YLKS/https%3A%2F%2Fhackmd.io%2F%40shaoeChen%2FSyez2AmFr](https://hackmd.io/@shaoeChen/Bywb8YLKS/https%3A%2F%2Fhackmd.io%2F%40shaoeChen%2FSyez2AmFr) 李宏毅RL课程
2. [https://arxiv.org/pdf/1301.2315.pdf](https://arxiv.org/pdf/1301.2315.pdf) Baseline 作用