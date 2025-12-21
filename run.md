# 简化版Causal-HAFE
python train_simplified.py --model simplified_causal_hafe --dataset semeval2014

# 原版Causal-HAFE  
python train_causal.py --model causal_hafe --dataset semeval2014


tmux new -s train


tmux attach -t train


tmux kill-session -t train

挂起
正确的“挂起”姿势是：

按下键盘上的 Ctrl + B 键。
song b+ctrl

按一下 D 键。

此时你会看到终端显示 [detached]，说明你已经安全退出了，后台程序依然在跑。


tmux ls