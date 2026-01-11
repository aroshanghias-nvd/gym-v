# Sphinx 环境开发指南

基于 PR #21 的实现和重构，本文档整理了完整的环境开发规范，用于指导后续环境的开发。

---

## 一、设计哲学

### 1.1 核心目标
- **统一接口**：所有环境使用相同的 Gymnasium 兼容接口
- **视觉+文本**：每个 Observation 同时包含图像和文本描述
- **可复现性**：通过 seed 确保完全确定性的问题生成
- **可扩展性**：通过继承和模板方法模式支持多种变体

### 1.2 Sphinx 环境的定位
Sphinx 环境是**程序化生成的视觉推理任务**：
- 单步回答（`max_episode_steps=1`）
- 无需外部数据集，问题完全由代码生成
- 每次 reset 生成唯一的新问题
- 答案从多个选项中选择（如 a-h）

---

## 二、环境类型分类

### 2.1 四种实现模式

| 模式 | 特点 | max_episode_steps | 示例 |
|------|------|-------------------|------|
| **A. 数据集驱动** | 从外部数据集加载问题 | 1 | ReasoningGym/Sudoku |
| **B. 包装外部库** | 封装 textarena 等游戏库 | 100 | TextArena/Sokoban |
| **C. 自实现游戏** | 从零实现游戏逻辑 | 200 | GameRL/Snake |
| **D. 程序化生成** | 代码生成视觉推理问题 | 1 | **Sphinx/*** |

### 2.2 Sphinx 环境的任务类型

目前已实现的任务类型：

| 任务 | 描述 | 核心能力 | 环境 ID |
|------|------|----------|---------|
| **TransformResult** | 给定原图和变换描述，选出正确的变换结果 | 空间变换理解 | `Sphinx/TransformResult-v0`, `Sphinx/TransformResultPoly-v0` |
| **SymmetryFill** | 给定有缺失的对称网格，选出正确的补全选项 | 对称性推理 | `Sphinx/SymmetryFill-v0`, `Sphinx/SymmetryFillPoly-v0` |
| **OddOneOut** | 8 个形状中 7 个是同一形状的变换，找出不同的那个 | 视觉辨别、变换不变性 | `Sphinx/OddOneOut-v0`, `Sphinx/OddOneOutPoly-v0` |
| **SequenceCompletion** | 根据变换模式补全序列的下一个形状 | 模式识别、归纳推理 | `Sphinx/SequenceCompletion-v0`, `Sphinx/SequenceCompletionPoly-v0` |

### 2.3 可扩展的任务类型（建议）

基于 Sphinx 数据集的其他任务类型：

| 任务 | 描述 | 核心能力 |
|------|------|----------|
| **Analogy** | A:B = C:? 的类比推理 | 关系映射 |
| **Counting** | 计数特定元素 | 视觉计数 |
| **SpatialRelation** | 判断空间关系 | 空间理解 |
| **MissingTiles** | 找出缺失图块的正确匹配 | 形状匹配 |

---

## 三、代码架构

### 3.1 继承结构

```
gym_v.Env (core.py)
│
├── SphinxTransformResultBaseEnv (transform_result.py)
│   ├── SphinxTransformResultEnv        # Grid 变体
│   └── SphinxTransformResultPolyEnv    # Polygon 变体
│
├── SphinxSymmetryFillBaseEnv (symmetry_fill.py)
│   ├── SphinxSymmetryFillEnv           # Grid 变体
│   └── SphinxSymmetryFillPolyEnv       # Icon 变体
│
├── SphinxOddOneOutBaseEnv (odd_one_out.py)
│   ├── SphinxOddOneOutEnv              # Grid 变体
│   └── SphinxOddOneOutPolyEnv          # Polygon 变体
│
├── SphinxSequenceCompletionBaseEnv (sequence_completion.py)
│   ├── SphinxSequenceCompletionEnv     # Grid 变体
│   └── SphinxSequenceCompletionPolyEnv # Polygon 变体
│
└── [新任务BaseEnv] (new_task.py)
    ├── [新任务Env]                     # Grid 变体
    └── [新任务PolyEnv]                 # Poly 变体
```

### 3.2 基类模板

每个任务类型应有一个抽象基类：

```python
class SphinxXxxBaseEnv(Env):
    """任务 Xxx 的抽象基类"""

    def __init__(self, option_size: int, padding: int, **kwargs: Any):
        super().__init__(**kwargs)  # 必须！传递 max_episode_steps
        # 共享参数
        self._option_size = option_size
        self._padding = padding
        # 状态变量
        self._composed_image: Image.Image | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """任务描述（共享）"""
        return dedent("""...""").strip()

    @abstractmethod
    def _generate_problem(self) -> ...:
        """生成问题（子类实现）"""
        pass

    @abstractmethod
    def _get_metadata(self) -> dict[str, Any]:
        """返回元数据（子类实现）"""
        pass

    @abstractmethod
    def _log_reset(self) -> None:
        """日志记录（子类实现）"""
        pass

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)  # 必须！初始化随机数生成器
        # 调用子类生成方法
        # 组合图像
        # 设置 oracle_answer
        self._log_reset()
        return Observation(...), info

    def inner_step(self, action: str):
        # 答案验证（共享逻辑）
        correct = self._check_answer(action)
        reward = 1.0 if correct else 0.0
        return Observation(...), reward, True, False, info

    def render(self) -> Image.Image:
        return self._composed_image
```

### 3.3 子类实现

```python
class SphinxXxxEnv(SphinxXxxBaseEnv):
    """Grid 变体实现"""

    def __init__(self, grid_size: int = 5, num_colors: int = 4, **kwargs: Any):
        super().__init__(**kwargs)
        self._grid_size = grid_size
        self._num_colors = num_colors

    def _generate_problem(self):
        return generate_random_grid(self.np_random, ...)

    def _get_metadata(self) -> dict[str, Any]:
        return {"grid_size": self._grid_size, "num_colors": self._num_colors}

    def _log_reset(self) -> None:
        logger.info(f"Reset Sphinx Xxx: grid_size={self._grid_size}")


class SphinxXxxPolyEnv(SphinxXxxBaseEnv):
    """Poly 变体实现"""

    def __init__(self, img_size: int = 300, num_points: int = 8, style: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._img_size = img_size
        self._num_points = num_points
        self._style = style
        self._current_style: str | None = None

    def _generate_problem(self):
        result = generate_random_polygon(self.np_random, style=self._style, ...)
        self._current_style = self._style if self._style else "random"
        return result

    def _get_metadata(self) -> dict[str, Any]:
        return {"num_points": self._num_points, "style": self._current_style}

    def _log_reset(self) -> None:
        logger.info(f"Reset Sphinx XxxPoly: style={self._current_style}")
```

---

## 四、视觉风格变体

### 4.1 两种基础风格

| 风格 | 特点 | 生成函数 | 难度控制参数 |
|------|------|----------|--------------|
| **Grid** | ARC 风格的彩色网格 | `generate_random_grid()` | `grid_size`, `num_colors` |
| **Poly** | 几何图形/多边形 | `generate_random_polygon()` | `num_points`, `style` |

**注意**：环境类层面不暴露 `difficulty` 参数。难度控制由上层逻辑通过调整 `grid_size`、`num_colors`、`num_points`、`style` 等参数来实现。

### 4.2 Grid 风格参数

```python
# 基础参数
grid_size: int = 5      # 网格大小 (N×N)
num_colors: int = 4     # 使用的颜色数量
cell_size: int = 40     # 每个格子的像素大小

# 难度控制
# - grid_size 越大，问题越复杂
# - num_colors 越多，视觉干扰越大
```

### 4.3 Poly 风格参数

```python
# 基础参数
img_size: int = 300         # 图像尺寸
num_points: int = 8         # 多边形顶点数（越多越复杂）
line_width: int = 3         # 线条宽度
grid_divisions: int = 8     # 背景网格分割数

# 风格控制（8 种，按视觉复杂度递增）
POLY_STYLES = [
    "outline",    # 1. 简单轮廓
    "filled",     # 2. 填充
    "nested",     # 3. 嵌套
    "striped",    # 4. 条纹
    "gradient",   # 5. 渐变
    "3d",         # 6. 3D 效果
    "composite",  # 7. 组合
    "pixelated",  # 8. 像素化
]

# Icon 风格（4 种，用于 SymmetryFillPoly）
ICON_STYLES = ["simple", "colored", "nested", "complex"]

# 难度控制
# - num_points 越多，形状越复杂
# - style 选择越靠后的风格，视觉复杂度越高
```

---

## 五、Utils 组织规范

### 5.1 文件结构

```
gym_v/envs/sphinx/
├── __init__.py              # 导出所有环境类
├── transform_result.py      # TransformResult 基类 + 两个变体
├── symmetry_fill.py         # SymmetryFill 基类 + 两个变体
├── odd_one_out.py           # OddOneOut 基类 + 两个变体
├── sequence_completion.py   # SequenceCompletion 基类 + 两个变体
├── [new_task.py]            # 新任务...
└── utils.py                 # 所有工具函数
```

### 5.2 Utils 函数分类

```python
# ===== 常量 =====
TRANSFORMS = [...]          # 8 种几何变换
GRID_COLORS = [...]         # 10 种颜色
POLY_STYLES = [...]         # 8 种多边形风格
ICON_STYLES = [...]         # 4 种图标风格
SEQUENCE_PATTERNS = {...}   # 序列变换模式

# ===== 变换函数（通用）=====
apply_transform(img, transform)  # 应用几何变换

# ===== Grid 生成 =====
generate_random_grid(rng, grid_size, num_colors, cell_size)
generate_symmetric_2x2_grid(rng, ...)  # 用于 SymmetryFill

# ===== Polygon 生成 =====
generate_random_polygon(rng, img_size, num_points, style, ...)
_style_outline(), _style_filled(), ...  # 风格实现

# ===== Icon 生成 =====
generate_random_icon(rng, img_size, style, ...)
generate_symmetric_2x2_icons(rng, ...)  # 用于 SymmetryFillPoly
_icon_simple(), _icon_colored(), ...    # 风格实现

# ===== 干扰项生成 =====
generate_extra_distractors(correct, all_cells, rng, num_extra)
_images_are_similar(img1, img2)
_add_asymmetric_marker(img, rng)

# ===== 图像组合 =====
compose_8_options(original, options, correct_idx, ...)           # TransformResult 用
compose_symmetry_fill_8_options(question, options, correct_idx, ...)  # SymmetryFill 用
compose_odd_one_out_8_options(options, odd_idx, ...)             # OddOneOut 用
compose_sequence_completion_image(sequence, options, correct_idx, ...)  # SequenceCompletion 用
```

### 5.3 添加新生成函数的规范

```python
def generate_xxx(
    rng: np.random.Generator,  # 必须！用于可复现性
    param1: int = default,
    param2: str | None = None,
    ...
) -> Image.Image:
    """生成 xxx 图像。

    Args:
        rng: numpy 随机数生成器，用于可复现性
        param1: 参数描述
        ...

    Returns:
        PIL Image
    """
    # 使用 rng 而非 random 模块
    value = rng.integers(0, 10)
    rng.shuffle(array)

    # 返回 PIL Image
    return img
```

---

## 六、注册规范

### 6.1 注册位置

所有注册在 `gym_v/envs/__init__.py`：

```python
# Sphinx environments
register(
    id="Sphinx/TransformResult-v0",
    entry_point="gym_v.envs.sphinx.transform_result:SphinxTransformResultEnv",
    max_episode_steps=1,
    kwargs=dict(
        grid_size=5,
        num_colors=4,
        cell_size=40,
        option_size=280,
        padding=20,
    ),
)
```

### 6.2 命名规范

- **ID 格式**: `Sphinx/{TaskName}[-Variant]-v{version}`
  - `Sphinx/TransformResult-v0` - Grid 变体（默认）
  - `Sphinx/TransformResultPoly-v0` - Poly 变体

- **类名格式**: `Sphinx{TaskName}[Variant]Env`
  - `SphinxTransformResultEnv`
  - `SphinxTransformResultPolyEnv`

### 6.3 不要在 kwargs 中添加注释

```python
# 正确
kwargs=dict(
    grid_size=5,
    num_colors=4,
)

# 错误
kwargs=dict(
    grid_size=5,  # 这是网格大小
    num_colors=4,  # 这是颜色数
)
```

---

## 七、答案格式规范

### 7.1 标准选项格式

```python
# 8 选项标准格式
labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

# 4 选项格式
labels = ["(a)", "(b)", "(c)", "(d)"]
```

### 7.2 答案验证逻辑

```python
def _check_answer(self, action: str) -> bool:
    """标准化答案验证"""
    action_clean = action.strip().lower()

    # 容错处理：支持 "a", "(a)", "A" 等格式
    if not action_clean.startswith("("):
        action_clean = f"({action_clean})"
    if not action_clean.endswith(")"):
        action_clean = action_clean + ")"

    return action_clean == self._oracle_answer.lower()
```

---

## 八、测试规范

### 8.1 必须测试的内容

每个环境至少需要以下测试：

```python
class TestSphinxXxx(unittest.TestCase):

    def test_basic(self):
        """基础功能：reset + step"""
        env = gym_v.make("Sphinx/Xxx-v0")
        obs, info = env.reset(seed=42)

        # 验证 Observation
        self.assertIsNotNone(obs.image)
        self.assertIsNotNone(obs.text)

        # 验证 oracle_answer
        oracle = info.get("oracle_answer")
        self.assertIsNotNone(oracle)

        # 验证正确答案得分
        obs, reward, terminated, truncated, info = env.step(oracle)
        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)

    def test_deterministic(self):
        """确定性：相同 seed 产生相同结果"""
        env = gym_v.make("Sphinx/Xxx-v0")

        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)

        self.assertEqual(info1["oracle_answer"], info2["oracle_answer"])

    def test_multiple_resets(self):
        """多次 reset 产生不同问题"""
        env = gym_v.make("Sphinx/Xxx-v0")
        answers = set()

        for _ in range(10):
            _, info = env.reset()
            answers.add(info["oracle_answer"])

        # 应该有多个不同答案
        self.assertGreater(len(answers), 1)
```

### 8.2 测试文件位置

```
tests/
├── test_sphinx.py           # Sphinx 环境测试
├── test_output_sphinx_*/    # 输出样例（可选）
```

---

## 九、添加新环境的完整流程

### Step 1: 设计任务

1. 明确任务目标（测试什么能力）
2. 确定选项数量（4/8）
3. 确定是否需要 Grid/Poly 两种变体

### Step 2: 实现生成函数

在 `utils.py` 中添加：
```python
def generate_xxx_problem(rng, ...):
    """生成 Xxx 任务的问题"""
    ...
```

### Step 3: 实现环境类

创建 `xxx_task.py`：
```python
class SphinxXxxBaseEnv(Env):
    """基类"""
    ...

class SphinxXxxEnv(SphinxXxxBaseEnv):
    """Grid 变体"""
    ...

class SphinxXxxPolyEnv(SphinxXxxBaseEnv):
    """Poly 变体"""
    ...
```

### Step 4: 更新导出

在 `sphinx/__init__.py` 添加：
```python
from gym_v.envs.sphinx.xxx_task import (
    SphinxXxxEnv,
    SphinxXxxPolyEnv,
)
```

### Step 5: 注册环境

在 `gym_v/envs/__init__.py` 添加：
```python
register(
    id="Sphinx/Xxx-v0",
    entry_point="gym_v.envs.sphinx.xxx_task:SphinxXxxEnv",
    max_episode_steps=1,
    kwargs=dict(...),
)

register(
    id="Sphinx/XxxPoly-v0",
    entry_point="gym_v.envs.sphinx.xxx_task:SphinxXxxPolyEnv",
    max_episode_steps=1,
    kwargs=dict(...),
)
```

### Step 6: 添加测试

在 `tests/test_sphinx.py` 的 `SPHINX_ENVS` 列表中添加新环境 ID

### Step 7: 验证

```bash
# 代码检查
uv run ruff check gym_v/envs/sphinx/ --fix
uv run ruff format gym_v/envs/sphinx/

# 运行测试
uv run python -m unittest tests.test_sphinx -v

# 手动验证
uv run python -c "
import gym_v
env = gym_v.make('Sphinx/Xxx-v0')
obs, info = env.reset(seed=42)
obs.image.show()
print(f'Answer: {info[\"oracle_answer\"]}')
"
```

---

## 十、关键要点速查

### 必须做的事

- [ ] `super().__init__(**kwargs)` 传递 kwargs
- [ ] `super().reset(seed=seed)` 初始化随机数
- [ ] 使用 `self.np_random` 而非 `random` 模块
- [ ] 实现 `inner_step()` 而非 `step()`
- [ ] `render()` 直接返回图像，不做 None 检查
- [ ] 在 info 中返回 `oracle_answer`

### 不要做的事

- [ ] 不在 kwargs 注释中添加说明
- [ ] 不在 `__init__.py` 添加 docstring
- [ ] 不重复实现相同逻辑（提取到基类）
- [ ] 不使用 `random.randint()` 等非确定性方法
- [ ] 不在环境类层暴露 `difficulty` 参数（由上层控制）

### 代码风格

- 双引号字符串
- 88 字符行宽
- 类型注解
- ruff 格式化
