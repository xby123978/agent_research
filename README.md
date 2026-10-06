# 个人深度研究与知识工程助手

> 输入研究主题，自动完成 **路径规划 → 文献检索 → 知识整理 → 报告生成 → 沉淀入库** 全流程，持续构建个人知识库。
>
> 完整落地当前 Agent 领域 7 项核心技术：规划 / 记忆 / Tools / 多 Agent / MCP / Skills / JEV。

| 项目信息 | |
|---|---|
| 文档版本 | V1.2 |
| Python 版本 | 3.11+ |
| 架构文档 | [architecture-design.md](architecture-design.md) · [Claude.md](Claude.md) |
| 部署方式 | 本地单机（Docker 可选） |

---

## 核心能力

### 七层分层架构

```
┌─────────────────────────────────┐
│      用户交互层 Gradio/CLI      │
├─────────────────────────────────┤
│    多Agent编排层 LangGraph      │  ← 规划模块、四角色Agent
├─────────────────────────────────┤
│    能力服务层 Skills引擎        │  ← 可复用场景化工作流
├─────────────────────────────────┤
│   MCP协议适配层 MCP Client      │  ← 统一工具接入总线
├─────────────────────────────────┤
│    智能决策层 大模型 + JEV      │  ← 生成用大模型，判断用JEV
├─────────────────────────────────┤
│   记忆与知识层 四层记忆体系     │  ← 向量+图数据库混合存储
├─────────────────────────────────┤
│      基础设施层 数据/API/文件   │
└─────────────────────────────────┘
```

### 七项技术全覆盖

| 技术点 | 落位模块 | 实现状态 |
|--------|----------|----------|
| 规划模块 | 多Agent编排层 - 研究规划师 | ✅ ReAct + Tree of Thoughts + 回退机制 |
| 记忆模块 | 独立记忆层 | ✅ 四层记忆架构（短期/工作/长期/偏好） |
| Tools 工具调用 | MCP协议层 + 工具集层 | ✅ 4 个 MCP Server + 11 个工具 |
| 多 Agent | 多Agent编排层 | ✅ 四角色（Planner/Collector/Engineer/Writer）+ LangGraph |
| MCP 协议 | MCP协议适配层 | ✅ 标准化能力接入总线 |
| Skills 技能 | Skills技能引擎 | ✅ 5 个 Skill + YAML flow 编排 |
| JEV 决策 | JEV智能决策层 | ✅ OpenJev 真实判别模型 + 分级执行策略 |

---

## 快速开始

### 1. 环境准备

```powershell
# 克隆项目
git clone <repo-url>
cd agent_research

# 安装依赖
pip install -r requirements.txt

# 可选：本地 JEV 判别模型（推荐）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

### 2. 配置环境变量

```powershell
cp .env.example .env
# 编辑 .env 填入 DeepSeek API Key
```

关键配置：

```ini
# 主推理模型（生成类任务）
LLM_API_KEY=your_deepseek_api_key_here
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# JEV 判别模型自动加载 OpenJev 0.8B（本地，无需配置）
# 无 transformers/torch 时降级到 DeepSeek API，再降级到规则评分

# 嵌入模型（可选 Ollama 或 sentence-transformers）
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_FALLBACK_MODEL=all-MiniLM-L6-v2
```

### 3. 启动服务

**方式 A：Gradio 交互界面（推荐）**

```powershell
python -m web.app
```

访问 http://127.0.0.1:7860

**方式 B：REST API 服务**

```powershell
python -m src.api.server
```

API 文档：http://127.0.0.1:8000/docs

### 4. 运行测试

```powershell
# 单元测试
python -m pytest tests/unit/ -v

# 集成测试
python -m pytest tests/integration/ -v

# 基准评测（10 个标准任务）
python -m pytest tests/benchmark/test_benchmark.py -v

# 阶段三模块验证
python tests/test_phase3.py
```

---

## 项目结构

```
agent_research/
├── src/
│   ├── agents/               # 四角色 Agent
│   │   ├── base.py           # Agent 抽象基类（safe_execute + 超时隔离）
│   │   ├── planner_agent_v2.py   # 研究规划师（ReAct + ToT）
│   │   ├── collector_agent.py    # 文献收集者（检索 + 筛选）
│   │   ├── engineer_agent.py     # 知识工程师（知识点抽取）
│   │   └── writer_agent.py       # 报告撰写者（生成 + 事实校验）
│   ├── orchestrator/         # 多 Agent 编排
│   │   ├── orchestrator.py   # LangGraph StateGraph 驱动
│   │   ├── shared_board.py   # 线程安全共享黑板
│   │   └── state_machine.py  # 任务状态机
│   ├── planning/             # 规划模块
│   │   └── planner.py        # 3 候选方案 × 2 轮重试 + JEV 择优
│   ├── memory/               # 四层记忆
│   │   ├── manager.py        # 记忆管理器（JEV 写入校验 + 语义去重）
│   │   ├── short_term.py     # 短期记忆（SQLite）
│   │   ├── working.py        # 工作记忆（图谱 + SQLite）
│   │   ├── long_term.py      # 长期记忆（三路混合召回 + 权重融合）
│   │   └── preference.py     # 偏好记忆（配置 + 向量）
│   ├── models/               # 智能决策层
│   │   ├── llm.py            # 主推理模型客户端
│   │   ├── jev.py            # JEV 决策引擎（OpenJev 主导 + 降级链）
│   │   └── embedding.py     # 嵌入模型
│   ├── skills/               # Skills 引擎
│   │   ├── base.py           # Skill 抽象基类
│   │   ├── literature_review.py    # 文献综述 Skill
│   │   ├── knowledge_card.py       # 知识卡片 Skill
│   │   ├── citation_manager.py    # 引用管理 Skill
│   │   ├── argumentation_skill.py # 论证构建 Skill
│   │   ├── review_skill.py         # 研究复盘 Skill
│   │   ├── skill_definitions.py   # YAML flow 加载器
│   │   └── definitions/           # 3 个 YAML Skill 定义
│   ├── mcp/                  # MCP 协议层
│   │   ├── client.py         # MCP 客户端
│   │   └── servers.py        # 4 个 MCP Server
│   ├── tools/                # 工具集层
│   │   ├── base.py           # 工具基类
│   │   ├── openalex_tool.py  # OpenAlex 学术检索
│   │   ├── crossref_tool.py  # CrossRef 元数据查询
│   │   ├── pdf_marker_tool.py    # PDF 解析（Marker→PyMuPDF 降级）
│   │   ├── filesystem_tool.py    # 文件系统操作
│   │   ├── knowledge_base_tool.py # 知识库查询
│   │   ├── csl_tool.py        # CSL 引用格式转换（5 种格式）
│   │   └── obsidian_tool.py  # Obsidian 同步
│   ├── api/                  # REST API
│   │   └── server.py         # FastAPI 服务
│   └── common/               # 公共组件
│       ├── data_models.py    # Pydantic 数据模型
│       ├── quality_gates.py  # 质量门限（硬门限 + JEV 软门限）
│       ├── fact_check.py     # 事实校验与幻觉防控
│       ├── observability.py  # 可观测性（指标 + 任务归档）
│       ├── config.py         # 配置管理
│       ├── logger.py         # 日志（loguru）
│       └── utils.py          # 工具函数
├── web/
│   └── app.py                # Gradio 交互界面
├── tests/
│   ├── unit/                 # 单元测试
│   ├── integration/          # 集成测试
│   ├── benchmark/            # 基准评测（10 任务）
│   ├── test_phase3.py        # 阶段三模块验证
│   └── test_openjev_load.py  # OpenJev 加载测试
├── config/
│   ├── settings.yaml         # 系统配置
│   ├── models.yaml           # 模型配置
│   └── mcp_servers.yaml      # MCP Server 配置
├── data/                     # 数据目录
│   ├── sqlite/               # SQLite 数据库
│   ├── working.db            # 工作记忆图谱
│   └── preferences.json       # 偏好配置
├── outputs/                  # 输出目录
│   ├── archive/              # 任务归档（JSON）
│   ├── reports/              # 研究报告
│   └── obsidian/             # Obsidian 同步笔记
├── logs/                     # 日志目录
├── mcp_servers/              # 独立 MCP Server 进程
├── architecture-design.md    # 架构设计文档
├── Claude.md                 # 开发规范
├── docker-compose.yml        # Docker Compose（可选）
├── requirements.txt          # 依赖清单
└── .env.example              # 环境变量示例
```

---

## 核心模块详解

### 多 Agent 协作（架构 §3.1）

四角色分工，状态机驱动，LangGraph 编排：

| Agent | 职责 | 输入 | 输出 |
|---|---|---|---|
| PlannerAgent | 任务拆解、路径规划、动态调整 | 研究主题 | 子任务列表 + 执行计划 |
| CollectorAgent | 文献检索、去重、筛选 | 子任务 | 文献元数据列表 |
| EngineerAgent | 知识点抽取、关系建模 | 文献列表 | 知识点 + 图谱边 |
| WriterAgent | 报告生成、事实校验、沉淀 | 知识点 + 文献 | 研究报告 |

**状态机流转**：

```
PENDING_PLANNING → PENDING_SEARCH → PENDING_FILTER → PENDING_READING
                                                       ↓
PENDING_OUTPUT ← PENDING_ORGANIZING ←───────（支持回退）
     ↓
COMPLETED / FAILED
```

**异常升级机制**：工人 Agent 异常自动升级到 PENDING_PLANNING 重规划（最多 2 轮）。

### 四层记忆（架构 §3.3）

| 层级 | 介质 | 用途 | 召回方式 |
|---|---|---|---|
| 短期记忆 | SQLite | 当前任务上下文 | 精确查询 |
| 工作记忆 | SQLite + 图谱索引 | 任务内知识图谱 | BFS 遍历 + 关键词 |
| 长期记忆 | Qdrant（本地持久化 / Server） | 历史研究成果 | 混合召回：向量60% + 关键词20% + 图谱20% |
| 偏好记忆 | 配置 + 向量 | 用户偏好 | 配置读取 |

**写入校验**：
- JEV 质量评分 < 0.3 拒绝写入
- 语义去重（相似度 > 0.92 跳过）
- 自动建立图谱关联

### JEV 决策引擎（架构 §3.6）

**主模式：OpenJev 0.8B 本地判别模型**

基于 Qwen3.5-0.8B NLI 交叉编码器，单次 forward pass 输出蕴含/矛盾/中立三类概率，0 token 生成，与主推理 DeepSeek 完全独立。

**分级执行策略**（架构 §3.6.1）：

| confidence | 行为 |
|---|---|
| ≥ 0.8 | 自动执行 |
| 0.5 - 0.8 | 提交大模型复核 |
| < 0.5 | 丢弃，大模型重新决策 |

**六类落地场景**（架构 §3.6.2）：

1. 工具路由选择（Choice）
2. 文献相关性评分（Scoring）
3. 学术质量评分（Scoring）
4. 知识点重要性评分（Scoring）
5. 是否深入子主题（Bool）
6. 记忆召回充分性（Bool）

**降级链**：OpenJev → DeepSeek API → 规则评分

### MCP 协议层（架构 §3.4）

4 个 MCP Server：

| Server | 工具 | 功能 |
|---|---|---|
| AcademicSearchServer | search_openalex, search_crossref | 学术文献检索 |
| DocumentProcessorServer | parse_pdf | PDF 解析（Marker→PyMuPDF） |
| FilesystemServer | read_file, write_file, list_dir | 文件系统操作 |
| KnowledgeBaseServer | query_knowledge, add_knowledge | 知识库查询 |

### Skills 引擎（架构 §3.5）

5 个 Skill，YAML flow + quality_gates 编排：

| Skill | 功能 | 质量阈值 |
|---|---|---|
| literature_review | 文献综述生成 | 0.7 |
| knowledge_card | 知识卡片抽取 | 0.7 |
| citation_manager | 引用管理 | 0.7 |
| argumentation | 论证构建（论点-论据-论证链） | 0.7 |
| review | 研究复盘（经验沉淀） | 0.7 |

### 质量保障（架构 §6）

**全链路质量门限**（架构 §6.1）：

| 阶段 | 门限 |
|---|---|
| Planner | JEV 评分 ≥ 0.75 |
| Collector | 召回数 ≥ 20 篇，精确率 ≥ 0.7 |
| Engineer | 知识点完整度 ≥ 90% |
| Writer | 报告长度 ≥ 5000 字，引用率 ≥ 80%，综合评分 ≥ 0.85 |

**事实校验与幻觉防控**（架构 §6.4）：

- 双重来源校验：核心结论 ≥ 2 篇独立文献支持
- 引用溯源：所有引用观点标注 DOI
- JEV 幻觉检测：一致性评分 < 0.9 触发重写
- 人工复核入口：关键结论支持标记校验

**可观测性**（架构 §6.5）：
- 任务成功率、平均耗时、Token 消耗
- 工具调用成功率
- JEV 评分准确率
- 任务完整归档（输入/中间过程/输出/质量评分）

---

## REST API

| 端点 | 方法 | 功能 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/api/v1/tasks` | POST | 创建研究任务 |
| `/api/v1/tasks/{id}` | GET | 查询任务状态 |
| `/api/v1/tasks` | GET | 列出所有任务 |
| `/api/v1/memory/search` | POST | 记忆召回 |
| `/api/v1/memory/graph` | GET | 知识图谱可视化 |
| `/api/v1/skills` | GET | 列出可用 Skill |
| `/api/v1/skills/{name}/execute` | POST | 执行指定 Skill |
| `/api/v1/tools` | GET | 列出可用工具 |

API 文档：http://127.0.0.1:8000/docs

---

## Gradio 交互界面

访问 http://127.0.0.1:7860

**四个标签页**：

1. **研究任务**：输入主题、深度、最大文献数，流式展示四 Agent 协作进度 + 文献列表 + 知识点 + 报告
2. **任务档案与 JEV 评分**：核心指标看板 + 最近 10 个归档任务列表
3. **记忆查询**：长期记忆召回测试
4. **知识图谱**：图谱可视化浏览

---

## 模型选型（架构 §7.2）

遵循「**生成任务用好 API，高频判断本地跑**」原则：

### 主推理模型（生成类任务）

| 推荐等级 | 模型 | 适用场景 |
|---|---|---|
| ✅ 首选 | DeepSeek-V3 API | 任务规划、中文文献处理、报告撰写 |
| ⭐ 备选 | GPT-4o Mini API | 英文文献解析、MCP 工具调度 |
| 💻 本地 | Qwen2.5-14B-Instruct | 简单任务、敏感内容（需 16G 显存） |

### JEV 决策模型（判断类任务）

| 推荐等级 | 模型 | 适用场景 |
|---|---|---|
| ✅ 主模式 | OpenJev 0.8B（Qwen3.5 NLI） | 本地判别，0 token，独立审查 |
| 🔄 降级 | DeepSeek API | OpenJev 不可用时 |
| 🛡️ 兜底 | 规则评分 | 无 API 无网络时 |

---

## Docker 可选依赖

架构 §7.1 推荐的 3 项容器化服务，**当前已提供降级方案，Docker 非必需**：

| 服务 | Docker 版 | 当前降级方案 | 性能差距 |
|---|---|---|---|
| Qdrant（向量库） | 容器化 Server 模式（多进程/大规模） | Qdrant 本地持久化模式（数据落盘，无需 Docker） | <10万条无差异 |
| Neo4j Community（图库） | 容器化 Cypher 查询 | SQLite + 内存图谱索引 | 缺多跳路径算法 |
| Grobid（PDF 元数据） | 容器化结构化解析 | PyMuPDF 本地文本提取 | 缺章节/表格识别 |

**启用 Docker 版**：

```powershell
docker-compose up -d
# 修改 .env 切换开关
```

---

## 测试

### 测试结构

```
tests/
├── unit/                # 单元测试
│   ├── test_data_models.py
│   ├── test_short_term.py
│   ├── test_long_term.py
│   ├── test_planner.py
│   └── test_mcp.py
├── integration/        # 集成测试
│   └── test_e2e.py
├── benchmark/           # 基准评测（10 标准任务）
│   └── test_benchmark.py
├── test_phase3.py       # 阶段三模块验证
└── test_openjev_load.py # OpenJev 加载测试
```

### 基准评测任务（架构 §12.1）

1. Transformer 架构演进与变体
2. 多模态大模型训练方法
3. 知识图谱构建方法综述
4. 强化学习在机器人控制中的应用
5. 大模型在安全工程中的应用
6. Agent 记忆系统优化方案
7. 检索增强生成（RAG）技术
8. 长上下文大模型优化
9. 联邦学习隐私保护方法
10. 小样本学习在计算机视觉中的应用

### 验收指标（架构 §9.1）

| 指标 | 目标 | 实测 |
|---|---|---|
| 端到端任务成功率 | ≥ 90% | ✅ 达标 |
| 文献检索 Recall@10 | ≥ 0.8 | ✅ 达标 |
| 记忆召回正常工作 | 是 | ✅ 达标 |
| 端到端报告完整度 | ≥ 0.85 | ✅ 达标 |
| 单任务平均耗时 | < 120s | ✅ 4.85s（CrossRef 主源）|

---

## 开发规范

详见 [Claude.md](Claude.md)。核心约定：

- **模块边界清晰**：七层分层，层间通过标准化接口交互
- **失败不阻塞主流程**：所有外部调用带异常隔离
- **JEV 独立于生成模型**：判别模型与生成模型分离，避免"自己评自己"
- **质量门限硬约束**：各阶段准出校验，不达标不流转
- **降级方案完备**：每项外部依赖都有本地降级路径

---

## 分阶段落地路线图（架构 §9）

### 阶段一：MVP ✅
- 四 Agent 流水线（手写循环）
- Gradio UI
- 基础记忆（SQLite）
- OpenAlex/CrossRef 检索

### 阶段二：核心能力 ✅
- LangGraph 编排
- 四层记忆 + 混合召回
- Skills 引擎（5 个 Skill）
- MCP 协议层（4 个 Server）
- REST API
- 质量门限 + 可观测性
- 基准评测集

### 阶段三：高级能力 ✅
- JEV 决策引擎（OpenJev 真实判别模型）
- 分级执行策略
- 事实校验与幻觉防控
- ReAct + Tree of Thoughts 规划
- 任务档案可视化

---

## License

MIT
