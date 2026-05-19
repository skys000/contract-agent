---
name: "劳动合同合规审查智能体"
description: "极简、安全、专业的双智能体协同合规审查系统（亮色模式）"
colors:
  primary: "#0071e3"
  neutral-bg: "#f5f5f7"
  neutral-card: "#ffffff"
  neutral-border: "#d2d2d7"
  text-primary: "#1d1d1f"
  text-secondary: "#86868b"
  risk-high: "#ff3b30"
  risk-medium: "#ff9500"
  risk-low: "#34c759"
typography:
  display:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    fontSize: "30px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.02em"
  body:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "normal"
rounded:
  sm: "6px"
  md: "12px"
spacing:
  sm: "8px"
  md: "16px"
  lg: "24px"
components:
  card:
    backgroundColor: "{colors.neutral-card}"
    rounded: "{rounded.md}"
    border: "1px solid {colors.neutral-border}"
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "#ffffff"
    rounded: "{rounded.sm}"
---

# Design System: 劳动合同合规审查智能体 (Apple Light Mode)

## 1. Overview

**Creative North Star: "The Silver Studio" (白银工作室)**

这套视觉系统基于苹果公司标志性的亮色（Light Mode）极简主义美学进行重塑。全局使用纯白、银灰与细微的冷灰色阶，配以高对比度的近黑（#1d1d1f）正文字体，彻底消除了低对比度、文本模糊等弱光阅读痛点。界面依靠严谨的留白与原生的 1px 薄银线容器进行隔离，传达出极致的严谨、公开、安全与法务专业信誉。

**Key Characteristics:**
- **极致高对比**：正文使用近黑 `#1d1d1f`，底色使用纯白 `#ffffff` 或淡银灰 `#f5f5f7`，文字边缘清晰。
- **Streamlit 原生组件风格改写**：摒弃不合规的 HTML 自闭合 `div` 切分卡片，转而使用 `st.container(border=True)` 作为容器，利用 CSS 改写原生边框。
- **零 AI Slop 渲染**：彻底移除顶部白条杂色、彩虹渐变和发光的毛玻璃背板。

## 2. Colors

### Primary
- **Apple Royal Blue** (#0071e3): 用于主行动按钮、操作提示和激活项的高亮显示。

### Neutral
- **Apple Light Gray** (#f5f5f7): 页面大背景色，营造洁净、现代的呼吸感。
- **Studio White** (#ffffff): 卡片、输入框与主要容器的背景色，高雅的纯白色。
- **Silver Divider** (#d2d2d7 / #e5e5ea): 用于 1px 细线边框与分隔线，是建立高级质感的核心。
- **Ink Black** (#1d1d1f): 主要文字与重要标题，高对比度，阅读舒适。
- **Muted Gray** (#86868b): 辅助文本、小字描述和失效状态文字。

### Named Rules
**The Non-Gradient Rule.** 页面内严禁使用任何彩色渐变（无论文字还是背景）。统一使用纯单色填充。
**The White Canvas Rule.** 页面背景以银灰（#f5f5f7）为基底，所有容器卡片背景必须为纯白（#ffffff），形成经典视差。

## 3. Typography

**Display Font:** Inter, -apple-system, system-ui
**Body Font:** Inter, system-ui

### Hierarchy
- **Display** (Bold, 30px, Line Height 1.2): 主页面大标题。
- **Headline** (SemiBold, 18px, Line Height 1.3): 卡片或分栏标题。
- **Body** (Regular, 14px, Line Height 1.5): 合同文本及合规报告正文。行宽控制在 75ch 以内。
- **Label** (Medium, 12px, Line Height 1.0): 徽章文本、小提示。

## 4. Elevation

卡片采用纯扁平或极弱阴影设计。

### Named Rules
**The Flat-Border Rule.** 容器在静态下无阴影，仅依靠 1px 的 `#d2d2d7` 银灰边框。悬停交互时仅允许产生极其柔和的弥散投影 `0 4px 16px rgba(0,0,0,0.04)`。

## 5. Components

### Cards / Containers
我们直接自定义 Streamlit 的原生 container 边框样式：
- **Border**: `1px solid #d2d2d7`
- **Background**: `#ffffff`
- **Border Radius**: `12px`

### Buttons
- **Primary Button**: 背景色 `#0071e3`，文字纯白，8px 圆角。
- **Secondary Button**: 背景色 `#ffffff`，边框 `1px solid #d2d2d7`，文字 `#1d1d1f`。

### Navigation
- **Top Tabs**: 苹果横向扁平胶囊。激活项为 `#ffffff` 背景并伴有弱阴影，背景条为 `#e5e5ea`，激活文字为 `#0071e3`。

### Audit Badge
- **Style**:
  - 高风险 (High): 背景为 `rgba(255, 59, 48, 0.12)` + 文字为 `#ff3b30`。
  - 中风险 (Medium): 背景为 `rgba(255, 149, 0, 0.12)` + 文字为 `#ff9500`。
  - 低风险 (Low): 背景为 `rgba(52, 199, 89, 0.12)` + 文字为 `#34c759`。

## 6. Do's and Don'ts

### Do:
- **Do** 保持全站的高对比度阅读体验，所有在白底或灰底上的正文文字必须使用近黑 `#1d1d1f`。
- **Do** 隐藏 Streamlit 顶部的白色与彩色默认 Header，保持极简白银质感。
- **Do** 使用原生的 `st.container(border=True)` 并在全局 CSS 中改写其样式，解决分裂 HTML 容器导致的空卡片问题。

### Don't:
- **Don't** 使用暗色卡片或暗色输入框。
- **Don't** 将黑色或暗灰文本放在深色容器内，造成无法阅读的尴尬。
- **Don't** 在卡片内嵌套其他卡片。
- **Don't** 使用任何文字渐变或侧边粗线条装饰。
