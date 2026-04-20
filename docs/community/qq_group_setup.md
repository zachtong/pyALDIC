# QQ 群搭建文案（中文用户社群）

本文件是**给维护者自己用的速查表**，记录建 QQ 群时需要粘贴的所有文案。不对外发布。

---

## 群名称（建议）

> `pyALDIC · AL-DIC 用户交流群`

## 群头像

用 `src/al_dic/gui/assets/icon/pyALDIC-256.png`。

## 群信息填写

| 字段 | 建议填 |
|---|---|
| 群分类 | 学习 → 学习研究 |
| 群地区 | 全国 |
| 群人数上限 | 500（足够用，不够再扩到 1000/2000） |
| 加群方式 | ✅ 需要身份验证（防广告号） |
| 身份验证问题 | `从哪里了解到 pyALDIC？（简答即可）` |
| 群标签 | `DIC`、`实验力学`、`数字图像相关`、`Python`、`开源` |

---

## 群公告（建好群后粘贴到群公告）

```
📌 pyALDIC 用户交流群 · 群公告

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

欢迎！本群是 pyALDIC（Augmented Lagrangian Digital Image
Correlation in Python）的中文用户交流群。

🔗 GitHub 仓库：
   https://github.com/zachtong/pyALDIC
📦 安装命令：
   pip install al-dic
📖 中文文档：README 内已集成（GUI 支持简体中文界面）
🎥 端到端演示：README 中的 78 秒视频

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📮 提问方式建议

1. 运行问题 / 使用问题 → 群内直接问，或去 GitHub Discussions
2. Bug 上报 → 去 GitHub Issues（附版本号 + 重现步骤）
3. 功能建议 → GitHub Discussions Ideas 分类
4. 合作、私下咨询 → 邮件 zachtong@utexas.edu

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ 群规

1. 讨论限于 DIC、实验力学、pyALDIC 使用及相关技术问题
2. 禁止广告、刷屏、商业推销（学术招聘和会议信息除外）
3. 涉及实验数据、未发表论文等敏感内容请私下交流
4. 提问前请先查群文件 / 已有聊天记录 / GitHub Discussions
5. 回答别人问题请用【@+昵称】+ 完整回答，方便后续搜索

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📂 群文件夹

- `getting_started/`  —— 入门图文
- `example_data/`     —— 示例散斑图
- `configs/`          —— 常用参数预设
- `papers/`           —— 相关论文（请自行核实版权）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

维护者：Zixiang Tong（佟子翔）@ UT Austin · Dr. Jin Yang 课题组
最后更新：2026-04-20
```

---

## 群精华消息（建好群后发几条并设为精华）

### 消息 1：怎么提一个好问题

```
🙋 怎么提一个好问题

1. 版本：在 Python 中跑一下：
   python -c "import al_dic; print(al_dic.__version__)"

2. 环境：Windows / macOS / Linux + Python 版本

3. 现象：你做了什么、期望什么、实际发生了什么

4. 重现：如果能提供最小图像序列或参数，最好

5. 日志：软件里的 LOG 面板或终端输出（截图或复制文本）

把这五项一并丢出来，答复效率会显著提升。
```

### 消息 2：v0.4.0 发布公告

```
🎉 pyALDIC v0.4.0 发布 —— 多语言支持（8 种）

GUI 现已完整支持：
🇨🇳 简体中文  🇹🇼 繁體中文  🇯🇵 日本語   🇰🇷 한국어
🇺🇸 English   🇩🇪 Deutsch   🇫🇷 Français  🇪🇸 Español

切换方式：顶部菜单 → 设置 → 语言。重启后生效。

v0.4.0 其他亮点：
• 端到端 78 秒演示视频（README 内嵌）
• README 大改版：对比表格客观化 + Accuracy 指向同行评议论文
  （Yang & Bhattacharya 2019、Tong et al. 2025、DIC Challenge 2.0）
• CJK 字体回退链：中文界面不再是宋体，微软雅黑/苹方一键适配

升级：pip install --upgrade al-dic

完整日志：https://github.com/zachtong/pyALDIC/blob/main/CHANGELOG.md
下载：https://github.com/zachtong/pyALDIC/releases/tag/v0.4.0
```

---

## 加入方式（放到 README 和用户指南里）

拿到群号 + 二维码之后这样放：

```markdown
🇨🇳 **中文用户 QQ 群**：群号 `XXXXXXXXX`

扫码入群（二维码 7 天有效，过期请联系维护者）：

<img src="assets/qq_group_qr.png" alt="pyALDIC QQ 群二维码" width="180"/>
```

---

## 二维码维护提醒

- QQ 群二维码**默认 7 天过期**
- 到期前手动打开 QQ 群 → 群设置 → "二维码" → 刷新 → 重新截图保存到 `assets/qq_group_qr.png` → 提交 commit
- 更省心的做法：群设置里选 "**关闭**搜索和二维码"，只用**群号**+身份验证问题作为入群入口，一劳永逸
