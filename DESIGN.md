---
name: "劳动合同合规审查智能体"
description: "极简、安全、专业的双智能体协同合规审查系统"
colors:
  primary: "#2f9eef"
  neutral-bg: "#161617"
  neutral-card: "#252526"
  neutral-border: "#424245"
  text-primary: "#f5f5f7"
  text-secondary: "#86868b"
  risk-high: "#ff453a"
  risk-medium: "#ff9f0a"
  risk-low: "#30d158"
typography:
  display:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    fontSize: "32px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.02em"
  body:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    fontSize: "15px"
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
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
---

# Design System: 劳动合同合规审查智能体

## 1. Overview

**Creative North Star: "The Space-Gray Sanctuary" (深空灰避难所)**

这套视觉系统汲取自苹果公司的极简主义工业设计美学，追求克制、精准与极高的数据可读性。我们废除了所有带有“AI Slop”色彩的炫目渐变、厚重的玻璃拟态特效以及凌乱的页面堆叠。系统通过精心设计的页面解耦、严谨的网格留白与超薄的高品质边框，营造出安全、私密且不可动摇的法务专业氛围。

**Key Characteristics:**
- **极端克制**：除必要的风险状态（红/橙/绿）外，全站几乎只使用深空灰色阶与精细度极高的浅灰细线描边。
- **页面解耦**：彻底摒弃在一个页面中塞满看板和审查区域的混乱排版，转而采用清晰的多 Tab 标签页工作流导航。
- **高对比排版**：利用字重（700 vs 400）与文字颜色（纯白 vs 苹果次级灰）进行清晰的层级区分。

## 2. Colors

我们采用克制的深色模式方案，摒弃了蓝紫渐变，采用更纯净的深灰色阶，确保用户长时间审查法条时不易产生视觉疲劳。

### Primary
- **Apple Blue** (#2f9eef / oklch(70% 0.16 230)): 用于主要的操作按钮、交互状态与激活导航项，保持极高的视觉精确度。

### Neutral
- **Space Gray Background** (#161617): 极暗灰背景，避免使用刺眼的纯黑 (#000000)，tint 偏向品牌冷灰。
- **Card Container Gray** (#252526): 容器与卡片背景色，较底层背景略浅，建立逻辑层级。
- **Fine Border Silver** (#424245): 用于极细的 1px 细线边框与分隔线，是建立现代高级感的核心。
- **San Francisco White** (#f5f5f7): 主文本颜色，高雅的柔白色。
- **Secondary Muted Gray** (#86868b): 辅助和说明文本颜色，标准的苹果次级灰色。

### Named Rules
**The Mono-Accenting Rule.** 页面中的主要高亮色（蓝色）占比必须在 5% 以内。除了主行动按钮和激活菜单，其余区域均退隐为灰度表现。
**The Non-Gradient Text Rule.** 严禁在标题或正文中使用任何文字渐变。所有文本必须为单色，纯白或灰色，通过粗细和大小体现重要性。

## 3. Typography

**Display Font:** Inter, -apple-system, BlinkMacSystemFont (系统原生字体)
**Body Font:** Inter, system-ui (极高识别率的无衬线字体)

### Hierarchy
- **Display** (Bold, 32px, Line Height 1.2): 仅用于大页面的主要标题，体现极简气势。
- **Headline** (SemiBold, 20px, Line Height 1.3): 用于各区域卡片的标题。
- **Body** (Regular, 15px, Line Height 1.5): 用于合同文本展现、审计报告正文。最大行宽限制在 70ch 以内。
- **Label** (Medium, 12px, Line Height 1.0): 用于状态徽章、操作提示、次要元数据说明。

## 4. Elevation

我们追求纯平或极轻量的视差美学，避免使用大范围厚重的模糊阴影。深度关系主要通过色彩阶梯（#161617 到 #252526）和细线边框传达。

### Named Rules
**The Flat-By-Default Rule.** 卡片在静态下保持扁平，无阴影，仅依靠 1px 的 `#424245` 银灰边框。只有在鼠标悬停或拖拽等交互响应时，才可产生极轻的扩散阴影 `0 4px 12px rgba(0,0,0,0.25)`。

## 5. Components

### Cards / Containers
- **Corner Style**: 12px 圆角 (Standard Apple Corner)。
- **Background**: `#252526`。
- **Border**: `1px solid #424245`。
- **Internal Padding**: 24px 宽裕内边距。

### Buttons
- **Shape**: 8px 圆角。
- **Primary Button**: 背景色为 `#2f9eef`，无边框，文字为纯白。
- **Secondary Button**: 透明背景，边框为 `1px solid #424245`，文字为 `#f5f5f7`。

### Navigation
- **Top / Sidebar Tabs**: 采用横向或纵向扁平胶囊式 Tab，激活项为微弱的浅色背景并加粗，未激活项淡化为 `#86868b`。

### Audit Badge
- **Style**: 对风险等级进行分类的徽章：
  - 高风险 (High): 背景为 `#ff453a` (30% 不透明度) + 文字为 `#ff453a`。
  - 中风险 (Medium): 背景为 `#ff9f0a` (30% 不透明度) + 文字为 `#ff9f0a`。
  - 低风险 (Low): 背景为 `#30d158` (30% 不透明度) + 文字为 `#30d158`。

## 6. Do's and Don'ts

### Do:
- **Do** 使用 Tab 标签页将“大屏看板”与“合同会审工作区”拆分，实现视觉上的大幅减负。
- **Do** 使用 `1px solid #424245` 的超细边框将各卡片模块界限分清，确保严谨与专业。
- **Do** 对脱敏部分（身份证/手机号）添加冷静的灰色高亮背景框，强调其安全性。

### Don't:
- **Don't** 使用侧边粗边框（Side-stripe borders）装饰警告或卡片。
- **Don't** 在任何地方使用背景渐变文字（`background-clip: text`）。
- **Don't** 在卡片内嵌套其他卡片。
- **Don't** 在未经用户请求的情况下使用浮夸的玻璃模糊（backdrop-filter: blur）做主要布局背景。
