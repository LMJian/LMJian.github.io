---
title: "REA 入门：让 Agent 带着证据理解软件，并接入 Codex 与 DeepSeek Harness"
description: "从关键词、交叉引用到证据链，理解 REA 的逆向分析工具链，接入 Codex 与 DeepSeek Harness，并用最小 JavaScript 案例验证边界。"
date: 2026-10-23 22:10:00 +0800
categories: [人工智能, AI工程]
tags: [REA, Agent, MCP, Codex, DeepSeek Harness, 逆向工程, 后端开发]
toc: true
media_subpath: /assets/img/posts/rea
---

维护一个没有原始工程的旧客户端时，你可能只知道一个现象：“点击导出后，文件名是怎样生成的？”手上没有源码仓库，只有安装包、网页脚本或一份可执行文件。

把这个问题直接交给大模型，模型可能讲出一套合理的实现，却没有证据说明目标软件真的这样运行。要回答实际行为，需要检查软件产物、追踪调用关系，必要时再做运行验证。

**REA，全称 Reverse Engineer Anything，是把专业逆向分析能力封装给 Agent 使用的开源工具。** 它帮助 Agent 调用分析工具、沿着功能线索调查，并把结论连接到可核查的证据。

本文先解释 REA 的工作方式，再给出 Codex 和 DeepSeek Harness 的接入配置，最后用一个自建的运费计算程序演示如何建立证据链。REA 命令固定使用 **`rea-agents@6.4.0`**；客户端配置参考 **2026 年 10 月 23 日可查的官方文档**。

<!-- download-package:start -->
<a id="download"></a>

[下载完整学习包（ZIP）]({{ '/assets/downloads/rea-demo.zip' | relative_url }})。包内包含离线 Markdown、3 张示意图、JavaScript 教学样本、配置片段与 Node 边界测试和复核记录。下文的本地运行命令以解压后的离线 Markdown 所在目录为工作目录。
<!-- download-package:end -->

## 1. 逆向分析解决什么问题

**逆向工程（Reverse Engineering）** 是从已有软件产物和运行现象出发，分析结构、接口和行为。对后端开发者来说，它类似于接手一个缺少设计资料的系统：先找入口，再追踪调用，最后用证据解释现象。区别在于，你面对的材料可能是机器码、压缩脚本或 APK，而不是完整源码。

几个术语先解释清楚：

| 术语 | 可以怎样理解 |
|---|---|
| 二进制程序 | 编译后的软件产物；原始变量名、注释和工程结构可能已经丢失 |
| 反汇编 | 将机器指令表示为汇编指令，便于检查底层操作 |
| 反编译 | 从编译产物恢复较易阅读的近似代码；通常无法还原原始源码 |
| 交叉引用，简称 xref | 某个字符串、函数或地址被哪些位置使用，用来追踪线索 |
| 静态分析 | 检查文件和代码结构，不必执行目标程序 |
| 动态分析 | 在受控条件下运行目标，观察具体场景的实际行为 |

REA 的用途是让 Agent 更有效地开展这类调查。它不能凭空补回编译时丢掉的信息，也不会因为工具名里有 “Anything” 就支持所有文件、平台和混淆方式。[REA 官方仓库](https://github.com/morluto/rea)

## 2. REA、MCP、Harness 分别做什么

这里的 **Agent（智能体）** 是根据目标和工具反馈反复选择下一步操作的程序；**Harness** 是承载这个执行循环的运行框架。

可以把整体理解成三层：

- **Agent / Harness**：模型理解问题、选择下一步，运行框架管理上下文、权限和工具调用。
- **REA**：提供分析入口，编排常见调查步骤，整理证据和分析结果。
- **专业分析后端**：真正解析不同软件产物或观察运行中的页面。

![REA 通过 MCP 连接 Agent 与不同专业分析后端](rea-toolchain.png){: width="1600" height="880" }

**MCP（Model Context Protocol，模型上下文协议）** 为客户端发现和调用外部工具提供统一接口。可以把它类比为工具层的标准接入协议。它不是模型，也不替代 Ghidra、JADX 等分析程序。

REA 6.4.0 随包的 Skill 元数据标注了 **170 个工具**。这个数字反映该版本的工具目录规模，不表示每个环境都已经具备全部分析能力。实际可见工具以 MCP 工具列表为准，实际可执行能力还取决于后端是否安装、配置和可用。

### 不同目标使用不同工具链

| 目标 | 主要分析路径 | 需要理解的边界 |
|---|---|---|
| 原生程序 | 对接 Hopper、Ghidra、IDA 等 | 平台、架构、符号信息和对应后端会影响分析质量 |
| 网页应用 | 通过 CDP 读取页面和脚本，观察浏览器侧信息 | 只能看到可访问的客户端材料与观察结果，不能直接读到服务端源码 |
| JavaScript / Electron 产物 | 静态分析脚本、模块及其关系 | 不一定需要原生反编译器；动态加载、混淆等仍会限制结论 |
| Android APK | 使用 JADX 分析 DEX 等内容 | 需要完整 JDK；原生库、资源、运行行为可能还需其他工具 |

**CDP（Chrome DevTools Protocol）** 是浏览器提供的调试协议。它能让工具获取页面、脚本等信息，但前提是有可连接、已授权的浏览器调试环境。接上 MCP 不会自动开启浏览器调试。

**JADX** 主要把 Android 的 DEX 字节码转为便于阅读的 Java 类代码。**JDK（Java Development Kit）** 是完整的 Java 开发工具包，与只提供运行环境的 JRE 不同。APK 中的原生 `.so` 库并不因此变成 Java 源码，可能需要走原生分析路径。[REA 官方功能与安装说明](https://github.com/morluto/rea#readme)

## 3. 从关键词到函数：把调查经验做成工具

假设你要调查“点击导出后如何生成文件”。直接让模型浏览全部反编译结果，既费上下文，也容易迷失。

一个更聚焦的调查顺序是：

1. 找与功能相关的字符串，例如菜单文字、错误提示或文件扩展名。
2. 查这些字符串被哪些函数引用。
3. 查看相关函数，再沿调用关系向前或向后追踪。
4. 检查数据从哪里来、条件怎样判断、结果传给哪里。
5. 对关键假设做验证，记录哪些路径仍然未知。

REA 的 **`trace_feature`** 就把“关键词 → 引用 → 相关函数”的常见调查步骤封装成工具。对于已经由相应原生分析后端打开的目标，其 6.4.0 输入包括：

```json
{"query": "Export failed", "case_sensitive": false}
```

这是一份工具参数示例，不是可以直接粘贴进终端的命令。它要求目标和分析上下文已经就绪，返回候选线索后还要检查实际引用和函数。不能把它理解为输入任意业务问题，就自动证明所有平台上的完整行为。

JavaScript 产物有对应的静态分析和功能追踪能力。使用时先看客户端当前暴露的工具及输入 **schema（模式定义）**，也就是字段名称、类型和约束，避免把原生工具的前提和参数套到其他目标上。

![从业务问题、关键词和引用，到函数关系与验证的证据链](evidence-chain.png){: width="1600" height="850" }

这里体现了 REA 的一个重要价值：把可重复的调查步骤标准化，让模型把注意力放在问题拆解和证据解释上。但步骤变得自动，并不会让结论自动正确。

## 4. Evidence：结论为什么必须能追溯

**Evidence（证据）** 是支撑结论的分析记录，例如目标文件的哈希、字符串位置、函数地址、调用关系或具体运行观察。**哈希**是根据文件内容计算出的指纹，用来标识分析的是哪份产物。保留这些记录，才能让另一个人回到同一个目标和位置，复查推理过程。

写分析结论时，至少分清三类信息：

| 类型 | 运费案例中的表述 | 能说明什么 |
|---|---|---|
| 事实 | 在该文件中看到 `totalCents >= 9900` 的比较 | 某个具体代码位置存在这个判断 |
| 推断 | 该判断可能实现“满 99 元免运费” | 还依赖金额单位、调用入口及覆盖条件 |
| 未知 | 服务端是否会重新计算运费 | 当前客户端材料没有回答这个问题 |

一个哈希值可以标识分析的是哪份文件，不能证明功能解释正确。一个函数里存在网络调用，也不能证明它在你关心的场景中已经执行。

### 为什么摘要和分页也很重要

模型的上下文容量有限，工具结果却可能包含大量字符串、函数和反编译内容。REA 的结果组织方式包括摘要及按需、分页读取，目的是先定位问题，再展开相关证据。

实际使用时，可以先看概览、数量、限制和证据标识，再针对少量相关函数读取详情。需要完整枚举某个范围时，应检查该范围的后续分页；只调查一个功能时，可以先过滤再展开。摘要被截断或第一页没有命中，都不能被当成“已查遍整个软件”。具体字段和读取工具以当前工具 schema 为准。

## 5. 接入前：先确认版本和依赖

REA 6.4.0 的 npm 包声明了以下 Node.js 范围：

```text
^22.19.0 || ^24.11.0 || >=26.0.0
```

也就是 22.x 需要至少 22.19.0，24.x 需要至少 24.11.0，或者使用 26.0.0 及以上版本；23.x 和 25.x 不在声明范围内。这是 **REA 6.4.0 的运行要求**，不要把它当成所有 Harness 的通用要求。[npm 6.4.0 版本元数据](https://registry.npmjs.org/rea-agents/6.4.0)

先检查：

```bash
node --version
npm --version
npx -y rea-agents@6.4.0 --help
```

`npx` 会在需要时下载并运行 npm 包。这里的 `-y` 同意 npm 下载提示，不等于批准 REA 修改客户端配置或安装分析后端；REA 自身的 `setup --yes` 是另一层确认。

固定 `6.4.0` 能减少 REA 自动升级导致的命令、工具和行为变化，但不会固定 Node、Harness、分析后端或所有间接依赖。需要严格复现时，还应记录这些版本和目标文件哈希。

原生分析需要选择并准备可用后端；Android 路径需要相应 JDK、JADX 等依赖。只做下面的 JavaScript 静态示例，可以先不安装 Hopper、Ghidra 和 IDA。

## 6. 接入 Codex：注册 MCP，再确认当前会话真的连上

以下命令需要已安装 Codex CLI；先记录版本，再注册本地 stdio MCP 服务。仅使用桌面客户端、未安装 CLI 时，也可以采用后面的 TOML 配置方式。

```bash
codex --version
codex mcp add rea -- npx -y rea-agents@6.4.0 mcp
codex mcp list
```

**stdio（标准输入 / 标准输出）** 表示 Codex 启动 REA 子进程，通过进程的输入输出流通信，无须额外开放 HTTP 端口。标准输出承载协议数据，调试日志应写到标准错误，避免破坏通信。

也可以合并下面的片段到默认的 `~/.codex/config.toml`，两种方式选一种即可。已有 `[mcp_servers.rea]` 时修改原配置，不要再添加重复段落。

```toml
[mcp_servers.rea]
command = "npx"
args = ["-y", "rea-agents@6.4.0", "mcp"]
```

项目级配置位于 `.codex/config.toml`，适用时还要满足 Codex 的项目信任要求。配置位置和注册命令参考 [Codex 官方 MCP 文档](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)。

### MCP 与 Skill 要分别理解

MCP 提供可调用的工具；**Skill** 提供调查流程、决策规则和使用说明。安装 Skill 不会自动注册 MCP 服务，但 Skill 也可以指导 Agent 通过命令行调用 REA，因此 CLI 工作流不必以 MCP 连接为前提。只有 MCP 没有 Skill 时，工具可以工作，但 Agent 不一定遵循你期望的调查步骤。

REA 6.4.0 的统一安装入口是 `setup`。如果希望同时配置 Codex 和安装匹配的 Skill，可以采用下面这条路径，先预览，再应用；它可以替代前面的手工注册路径：

```bash
npx -y rea-agents@6.4.0 setup --client codex --dry-run
npx -y rea-agents@6.4.0 setup --client codex --yes
```

执行第二条前，检查预览中的文件位置和改动范围。**这个版本没有 `setup-skill` 子命令**，不能用它安装 Skill。不要再把新版本仓库中的 Skill 与固定的旧版服务混用，除非已经核对兼容性。[REA 官方安装说明](https://github.com/morluto/rea/blob/main/docs/installation.md)

保存配置或安装完成后，按客户端提示重新连接或重启会话。`codex mcp list` 只能证明存在注册信息，当前会话还应能够列出 REA 工具，并成功调用所需能力。

## 7. 接入 DeepSeek Harness：在正确的 patch 层注册服务

DeepSeek Harness 是另一套 Agent 运行框架。这里不是把 REA 直接接到“DeepSeek 模型参数”中，而是通过 Harness 的 MCP 客户端插件，让模型能使用 REA 工具。

本节假设 DeepSeek Harness 已安装，并且所选 profile 能正常启动；可以先用 `dsh --version` 记录版本。下面以已有的 `web` profile 为例。**Profile** 是一套独立运行配置，通常包含插件依赖和用户修改。默认用户目录下，它的 patch 文件是：

```text
~/.dsh/profiles/web/cordis.patch.yml
```

若要对本机所有 profile 应用同一修改，可以使用全局层：

```text
~/.dsh/cordis.patch.yml
```

若设置了 `DSH_HOME`，需要按实际根目录调整。选择符合预期的作用范围即可，不要在多个层里重复插入相同 `id`。[官方 patch 使用说明](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/mcp-memory.md)

把下面内容**合并**到现有 patch，而不是覆盖整个文件。示例包含 patch 需要的 `insert` 外层：

```yaml
- insert:
    - id: mcp-rea
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: rea
        transport: stdio
        command: npx
        args: ['-y', 'rea-agents@6.4.0', 'mcp']
```

`serverName: rea` 是工具命名空间。连接成功后，工具会以类似 `mcp__rea__trace_feature` 的名称出现。[官方 MCP 客户端配置](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/mcp/mcp-client/README.md)

当前官方 CLI 已随包提供 `@deepseek-ai/dsh-mcp-client` 依赖，常规安装下添加上述服务配置即可，不需要再安装同名插件。自定义宿主或裁剪安装若报模块找不到，则应按对应 Harness 版本补齐依赖。这个依赖并不等于已经启用某个 MCP 服务，每个服务仍需单独配置。[官方 CLI 依赖说明](https://github.com/deepseek-ai/deepseek-harness/blob/master/apps/cli/reference/README.md)

不要直接把生成的 `cordis.yml` 当成持久修改入口，也不要在不检查已有内容的情况下复制整个别人的配置。

### 给 DeepSeek Harness 安装匹配的 REA Skill

REA 6.4.0 可以单独更新 shared Skill 目录。这个操作不负责替你注册 DeepSeek Harness 的 MCP 插件：

```bash
npx -y rea-agents@6.4.0 setup --skill-client shared --dry-run
npx -y rea-agents@6.4.0 setup --skill-client shared --yes
```

先检查计划是否仅涉及你预期的 Skill 及相关检查。shared 默认目标是 `~/.agents/skills`；DeepSeek Harness 的文件系统 Skill 提供器可以扫描共享目录，但实际扫描根目录仍由当前 profile 的配置决定。安装后应确认会话目录里出现 `reverse-engineer-anything`；如果没有，检查 Skill 提供器和扫描路径。[官方 Skill 文件系统说明](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/skill/skill-filesystem/README.md)

修改后，支持热更新的配置可以重新加载连接；若工具或 Skill 没有刷新，重启当前 Harness profile，再检查加载日志和工具列表。不要根据聊天页面能打开，就认定 REA 已经可用。

## 8. 用四层检查判断接入是否成功

![配置、协议、分析能力和业务结论分别需要验证](integration-checks.png){: width="1600" height="900" }

| 检查层 | 要找的证据 | 常见失败原因 |
|---|---|---|
| 配置 | 当前客户端读取了预期文件和 profile | 改错目录、同名条目冲突、插件依赖缺失 |
| MCP 连接 | 初始化成功，工具发现完成 | `npx` 不在启动环境 PATH、下载失败、协议输出被污染 |
| 分析能力 | 小目标的所需工具调用成功 | 后端未安装、路径不对、许可未处理、目标格式不支持 |
| 业务结论 | 位置、调用关系与验证支持结论 | 把候选引用当最终答案、遗漏分支或服务端逻辑 |

第一次使用 `npx` 可能需要下载包，桌面客户端的 PATH 也可能与终端不同。排查时先使用客户端实际配置的命令在相同环境下检查，区分下载问题、服务进程问题和后端问题。

DeepSeek Harness 的官方 MCP 客户端允许部分连接失败时主程序继续启动，所以“程序启动成功”和“REA 连接成功”尤其需要分开确认。配置中的工具列表也不能替代一次真实的分析调用。

## 9. 最小案例：调查“满多少免运费”

用自建代码练习，可以把注意力放在分析方法上。示例目录是 `examples/owned-app/`，其中只有一个极小的 JavaScript 模块和包信息。分析时把这个目录当作收到的软件产物，不预先依赖设计说明。

JavaScript 文件本身仍是可读代码，这个练习主要展示调查和验证流程，也便于你对照代码检查结论。业务示例已运行验证；REA 分析命令按 6.4.0 发布物核对，完整 REA 工具链尚未实测通过，执行时应以实际返回的 Evidence 为准。原生二进制中的信息丢失、混淆等问题，需要另外的样本验证。

### 第一步：先提出可以验证的问题

目标问题是：**返回 `SHIPPING_QUOTE` 的逻辑，怎样计算运费？金额单位和边界条件是什么？**

用固定版本执行静态分析，把结果保存到文件：

```bash
npx -y rea-agents@6.4.0 analyze-javascript-application ./examples/owned-app --json > analysis.json
```

这条命令分析本地 JavaScript 产物，不需要浏览器或原生分析后端。先检查退出码及 JSON 中是否包含错误；读取统计、限制及图结构时，避免把完整输出一次性贴进模型上下文。

连接好 Agent 后，可以给它这样的任务：

```text
分析 examples/owned-app，只回答 SHIPPING_QUOTE 的运费计算逻辑。
先识别产物类型，使用适合 JavaScript 的静态分析能力。
从字符串定位到函数和调用关系，再检查阈值两侧的结果。
每项结论附文件或工具返回的证据位置，分开写事实、推断、未知。
不把客户端结论推广到未检查的服务端，不修改分析目标。
```

这段提示说明了验收标准。实际调用哪些工具、参数是什么，应依据当前服务公布的 schema；不需要强制让它把原生 `trace_feature` 用在 JavaScript 目录上。

### 第二步：沿引用检查计算逻辑

这个小程序的关键实现如下，完整文件见 [app.js](#download)：

```javascript
export function shippingFee(totalCents) {
  if (!Number.isSafeInteger(totalCents) || totalCents < 0) {
    throw new Error("INVALID_TOTAL");
  }
  return totalCents >= 9900 ? 0 : 800;
}

export function quote(totalCents) {
  const shipping = shippingFee(totalCents);
  return { label: "SHIPPING_QUOTE", totalCents, shipping };
}
```

字符串帮助定位 `quote`，调用关系带到 `shippingFee`。再观察到三个具体事实：输入必须为非负安全整数；阈值为 `9900`，包含等于；阈值以下返回 `800`。这里使用 `Number.isSafeInteger` 排除无法保证整数精度的输入；JavaScript 的安全整数上限是 `2^53 - 1`。用整数“分”表达金额，也不意味着可以忽略数值范围。

`totalCents` 的命名支持“金额以分为单位”的解释，但对陌生软件，仅凭名称还不够。真实调查应继续检查调用方如何转换金额、展示层如何格式化，或者对照一笔已知订单。这个自建示例约定单位为分，因此可以解释为满 99 元免运费，否则收 8 元。

### 第三步：验证边界，并限定结论范围

随包提供的 [check-demo.mjs](#download) 使用 Node 标准库验证阈值两侧，并检查负数、小数和超过安全整数范围的输入：

```javascript
import assert from 'node:assert/strict';
import { quote } from './owned-app/app.js';

assert.equal(quote(9899).shipping, 800);
assert.equal(quote(9900).shipping, 0);
for (const invalid of [-1, 0.5, Number.MAX_SAFE_INTEGER + 1]) {
  assert.throws(() => quote(invalid), /INVALID_TOTAL/);
}

console.log('通过：运费阈值、负数、小数和安全整数边界检查');
```

下载学习包并解压后，在离线 Markdown 所在目录运行完整文件：

```bash
node examples/check-demo.mjs
```

这验证的是本地示例函数在五个输入下的行为。它不是 REA MCP 连接测试，也没有证明真实电商系统的服务端计费规则。

一个合格的结论可以这样写：

> 在这份示例的 `app.js` 中，`quote` 调用 `shippingFee`。非负安全整数输入大于等于 9900 时，运费返回 0，否则返回 800。示例约定单位为分；测试验证了 9899、9900，以及负数、小数和超过安全整数范围的输入。本次没有涉及服务端重新计费、地区规则或优惠叠加。

这个表述能把代码事实、单位前提和验证范围连接起来。即使换成更复杂的软件，调查原则仍然相同。

## 10. 常见误区、限制与使用边界

**“反编译结果就是原始源码。”** 编译会丢失注释、部分名称和结构，优化还会合并或改写逻辑。反编译通常给出近似表示，需要结合汇编、引用和运行结果理解。

**“工具返回 Evidence，结论就一定正确。”** Evidence 让结论可追溯，仍需要核对目标版本、路径、推理与覆盖范围。过期文件、错误入口和未执行的分支，都可能影响解释。

**“在客户端没找到，就说明服务端没有。”** 客户端只能提供其可见范围内的证据。远程配置、动态代码、服务端判断或原生扩展可能位于其他地方。

**“170 个工具就是 170 项开箱即用的能力。”** 注册、协议、后端与目标格式是不同层。先用小样本验证当前任务真正需要的能力，再处理大目标。

**“REA 本地分析，所以资料一定不出本机。”** REA 的分析可以在本地进行，但工具结果会进入 Agent；若使用远程模型，结果可能按客户端和模型服务的策略发送出去。敏感目标需要先检查这条数据链路。

REA 的价值在于调用专业工具、复用调查流程、组织可追溯证据。应在授权范围内分析目标，核对目标许可与所用分析器许可。项目说明明确排除绕过 DRM、破解授权、制作外挂或复制专有代码等用途；其中 **DRM（Digital Rights Management，数字版权管理）** 是控制数字内容访问和使用的一类技术。REA 自身采用 MIT 许可，并不会替你授予分析其他软件的权利。[项目使用边界](https://github.com/morluto/rea/blob/main/DISCLAIMER.md)、[项目许可](https://github.com/morluto/rea/blob/main/LICENSE)

当你把“模型认为如此”改成“这个版本、这个位置、这条路径与这些测试支持这个结论”，Agent 才真正成为可协作的分析工具。REA 提供的是专业工具、标准化流程与模型推理的组合，分析质量最终仍取决于问题、证据和验证。
