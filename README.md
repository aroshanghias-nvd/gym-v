# gym-v

## 开发前可以先安装一下pre-commit (uv run pre-commit install)

## 阶段一

我们先给一些已经有的env集成进来，新增的env需要：
- 和原仓库的env的游戏行为一致
- 代码也尽量和原仓库保持一致（方便review）
- 如果原仓库是纯文本env，集成进来需要新增出图函数并且可以注意一下game description部分不要再包含原来的与纯文本caption / maze相关的内容
