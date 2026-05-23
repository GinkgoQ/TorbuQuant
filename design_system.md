# GinkgoQ Design System

![GinkgoQ favicon](../public/favicon.svg){ width="36" align=right }

This document defines the visual system for GinkgoQ publications, docs, research notes, API references, and MkDocs Material pages. Use it as the source of truth when adding pages, writing long-form content, or styling generated API documentation with `mkdocstrings`.

!!! note "Design intent"
GinkgoQ should feel like a precise deep-tech publication: calm, technical, signal-rich, and restrained. The visual language uses a dark/light editorial shell, ginkgo green as the only primary accent, crisp typography, thin borders, and generous spacing.

## Brand

### Name

The brand is written as `GinkgoQ` in prose and `GINKGOQ` in compact wordmarks.

The final `Q` is always green.

```html
Ginkgo<span class="brand-q">Q</span> GINKGO<span class="brand-q">Q</span>
```

In the Astro site this is handled by `src/components/BrandText.astro` and the global `.brand-q` rule.

### Mark

Use the ginkgo favicon beside the wordmark in navigation, footer, and documentation headers.

| Asset             | Path                  | Usage                                          |
| ----------------- | --------------------- | ---------------------------------------------- |
| Primary favicon   | `public/favicon.svg`  | Browser favicon, docs logo, header/footer mark |
| Alternate favicon | `public/favicon_.svg` | Fallback/alternate icon                        |

Rules:

- The logo mark uses `#79863c`.
- The mark should sit beside `GinkgoQ` with 8-10px visual spacing.
- Header mark size: about `30px`.
- Footer/docs mark size: about `28-30px`.
- Do not recolor the logo per page category.
- Do not place the logo inside a decorative badge unless the badge is part of navigation or metadata.

For MkDocs Material:

```yaml
theme:
  name: material
  logo: assets/favicon.svg
  favicon: assets/favicon.svg
```

Copy `public/favicon.svg` into `docs/assets/favicon.svg` when publishing MkDocs docs, or configure your docs build to reference the public asset.

## Color

### Primary Accent

The design theme color is ginkgo green.

| Token                  | Value                                           | Use                                 |
| ---------------------- | ----------------------------------------------- | ----------------------------------- |
| `--color-theme-green`  | `#79863c`                                       | Brand Q, logo mark, primary accents |
| `--color-accent`       | `#79863c`                                       | Links, active states, section rules |
| `--color-accent-hover` | `#667231` light / `#9baa60` dark                | Hover emphasis                      |
| `--color-accent-soft`  | `#edf1df` light / `rgb(121 134 60 / 0.18)` dark | Soft fills                          |
| `--color-accent-muted` | `rgb(121 134 60 / 0.12)`                        | Glows, quiet backgrounds            |

The accent is not decorative wallpaper. Use it for semantic emphasis: active navigation, links, badges, process icons, short rules, and the final `Q`.

### Light Theme

| Role              | Value     |
| ----------------- | --------- |
| Page background   | `#f8f8f5` |
| Main surface      | `#ffffff` |
| Secondary surface | `#fafaf8` |
| Tertiary surface  | `#f3f3ef` |
| Text              | `#101114` |
| Muted text        | `#5c6070` |
| Subtle text       | `#8e929e` |
| Border            | `#d8d8cf` |

### Dark Theme

| Role              | Value     |
| ----------------- | --------- |
| Page background   | `#0d0f15` |
| Main surface      | `#14161e` |
| Secondary surface | `#1a1d28` |
| Tertiary surface  | `#222638` |
| Text              | `#eceef8` |
| Muted text        | `#a1a6bc` |
| Subtle text       | `#747b98` |
| Border            | `#2a2f43` |

### MkDocs Palette

Use Material's custom CSS for exact token matching. Keep the built-in palette close to the brand:

```yaml
theme:
  name: material
  palette:
    - scheme: default
      primary: custom
      accent: custom
      toggle:
        icon: material/weather-night
        name: Switch to dark mode
    - scheme: slate
      primary: custom
      accent: custom
      toggle:
        icon: material/weather-sunny
        name: Switch to light mode
extra_css:
  - stylesheets/ginkgoq.css
```

Then define the exact colors in `docs/stylesheets/ginkgoq.css`:

```css
:root {
  --md-primary-fg-color: #79863c;
  --md-accent-fg-color: #79863c;
  --md-default-bg-color: #f8f8f5;
  --md-default-fg-color: #101114;
}

[data-md-color-scheme="slate"] {
  --md-primary-fg-color: #79863c;
  --md-accent-fg-color: #79863c;
  --md-default-bg-color: #0d0f15;
  --md-default-fg-color: #eceef8;
}
```

## Typography

### Fonts

| Role | Font                                                       |
| ---- | ---------------------------------------------------------- |
| Sans | `Inter`, `ui-sans-serif`, `system-ui`, `sans-serif`        |
| Mono | `IBM Plex Mono`, `SFMono-Regular`, `Consolas`, `monospace` |

Use Inter for prose and UI. Use IBM Plex Mono for metadata, labels, category tags, process step labels, keyboard hints, and code-adjacent UI.

### Headings

Rules:

- Large editorial headings use tight line-height.
- Avoid excessive negative tracking in compact UI.
- Keep text balanced and readable on mobile.
- Do not use hero-scale type inside cards, sidebars, or small panels.

Recommended scales:

| Element    | Size                                             |
| ---------- | ------------------------------------------------ |
| Hero H1    | `clamp(2.2rem, 4.6vw, 5.5rem)` depending on page |
| Page H1    | `clamp(3rem, 5.2vw, 5.5rem)`                     |
| Section H2 | `1.45rem - 2rem`                                 |
| Card title | `1rem - 1.14rem`                                 |
| Body       | `0.95rem - 1.05rem`                              |
| Metadata   | `0.72rem - 0.84rem`                              |

### Prose

Long-form content uses `.prose` with a maximum width of `720px`.

```css
.prose {
  max-width: 720px;
}

.prose h1,
.prose h2,
.prose h3 {
  line-height: 1.15;
  letter-spacing: -0.035em;
}
```

## Layout

### Shell

The website sits inside a card-like shell:

| Token              | Value                           |
| ------------------ | ------------------------------- |
| Container          | `1120px`                        |
| Outer page padding | `32px 24px` desktop             |
| Shell radius       | `14px`                          |
| Shell border       | `1px solid var(--color-line)`   |
| Shell shadow       | `0 24px 80px rgb(0 0 0 / 0.09)` |

Avoid painting full-width rectangular backgrounds inside a constrained `.container` unless the element has rounded corners and inset spacing. Otherwise it creates visible "cut" edges.

!!! warning "Avoid hard rectangular strips"
Hover states and bottom bands must not paint a square block across only the container width. Use inset rounded surfaces, full-width sections, or transparent borders instead.

### Containers

Use:

```css
.container {
  width: min(var(--container), calc(100% - 104px));
  margin: 0 auto;
}
```

On mobile, reduce horizontal padding per page when needed, but preserve readable line lengths.

## Components

### Header

Header behavior:

- Sticky at the top.
- Surface color equals `--color-surface`.
- Border appears on scroll.
- Brand mark uses green favicon mask.
- `GINKGOQ` wordmark has green `Q`.
- Desktop includes nav, theme toggle, search, RSS.
- Mobile simplifies to brand plus icon actions.

Header mark:

```css
.brand-mark {
  width: 30px;
  height: 30px;
  background: var(--color-theme-green);
  -webkit-mask: url("/favicon.svg") center / contain no-repeat;
  mask: url("/favicon.svg") center / contain no-repeat;
}
```

### Footer

Footer behavior:

- Same surface as the shell.
- Top border separates it from content.
- Bottom row uses only a border, not a separate constrained background.
- Social icons are quiet by default and use accent states on hover.

Do not add a separate footer-bottom background inside `.container`; it creates clipped-looking side blocks.

### Cards And Rows

Use cards only for repeated items, tools, modals, badges, and intentionally framed panels.

Rows such as latest posts should use an inset hover surface:

```css
.row {
  position: relative;
}

.row::before {
  content: "";
  position: absolute;
  inset: 0 -1.2rem;
  border-radius: 8px;
  opacity: 0;
  background: var(--color-surface-2);
  border: 1px solid var(--color-line);
}

.row:hover::before {
  opacity: 1;
}
```

### Badges

Badges can use brand-specific colors when linking to external platforms.

GitHub:

- Icon: GitHub mark.
- Primary color: `#24292f`.
- Secondary neutral: `#6e7681`.

Hugging Face:

- Icon: Hugging Face face mark or simple face approximation.
- Primary color: `#ffb000`.
- Soft fill: `rgb(255 176 0 / 0.16)`.

Badges should include:

- Icon block.
- Strong platform label.
- Short action text.
- Subtle brand-colored border.
- Hover lift no more than `2px`.

### Code Blocks

Code blocks use dark surfaces in both themes:

| Token           | Light     | Dark      |
| --------------- | --------- | --------- |
| Code background | `#13141d` | `#0d0e18` |
| Code border     | `#1e2033` | `#1c1e30` |
| Code text       | `#c8d0f5` | `#c0caf5` |

Copy buttons use the standard small radius and surface colors.

## Visual Assets

### Home Signal Artwork

Use:

- `/images/home/light.png` in light mode.
- `/images/home/dark.png` in dark mode.

The PNGs are not transparent. Keep them behind text and do not place them as small rectangles on a mismatched background.

Rules:

- Desktop: use the art as a background visual, small enough that the leaf remains in-frame.
- Mobile: soften opacity and hide competing animated grid layers if readability suffers.
- Preserve the halo/ambient glow as a separate overlay rather than relying only on the PNG.

### Process Icons

Use these assets:

- `/images/home/sense.svg`
- `/images/home/structure.svg`
- `/images/home/reason.svg`
- `/images/home/act.svg`

Render them through CSS masks when possible so they inherit `--color-accent`.

```css
.step-icon {
  background: var(--color-accent);
  -webkit-mask: var(--icon-url) center / contain no-repeat;
  mask: var(--icon-url) center / contain no-repeat;
}
```

## Motion

Motion should be subtle and technical:

- Use `--ease: cubic-bezier(0.4, 0, 0.2, 1)`.
- Use animated signal dots/fibers only where they support the concept.
- Reduce or hide motion on mobile when it competes with reading.
- Respect `prefers-reduced-motion`.

```css
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

## Publication Rules

Use these rules for every blog, research note, documentation page, and API reference page.

### Page Structure

Recommended order:

1. Title.
2. Short lead paragraph.
3. Metadata row if relevant.
4. Body content.
5. Related links or next steps.

### Writing Presentation

- Keep prose width near `720px`.
- Use short section headings.
- Use category labels in mono uppercase.
- Use green only for meaningful emphasis.
- Use tables for comparison, tokens, config, and API matrices.
- Use admonitions for warnings, caveats, design intent, and implementation notes.

### MkDocs Material Admonitions

Recommended usage:

```md
!!! note "Design intent"
Explain the reasoning behind a design or API choice.

!!! tip "Implementation"
Provide a concrete snippet or workflow.

!!! warning "Avoid"
Call out patterns that cause inconsistent publication design.
```

### mkdocstrings Pages

For generated API docs:

- Keep object headings compact.
- Avoid large hero treatments on API reference pages.
- Use code blocks and tables as the primary structure.
- Keep source links visible but visually quiet.
- Use the same green accent for active tabs, anchors, and selected nav items.

Suggested MkDocs config:

```yaml
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            show_root_heading: true
            show_root_full_path: false
            heading_level: 2
            members_order: source
```

## Accessibility

Requirements:

- Focus rings use `2px solid var(--color-accent)`.
- Interactive targets should be at least `34px` square in compact UI.
- Do not rely on green alone to communicate state.
- Keep contrast high in dark mode.
- Preserve semantic headings and landmarks.
- Use `aria-label` for icon-only buttons.

## Do And Do Not

Do:

- Use `#79863c` consistently for brand identity.
- Keep backgrounds continuous across section edges.
- Use rounded inset hover states for rows.
- Use platform-appropriate colors for external brand links.
- Keep mobile layouts simpler than desktop layouts.

Do not:

- Use blue or purple as the main accent.
- Put full-width rectangular strips inside a constrained container.
- Place text directly over high-detail art without an overlay.
- Use large card-heavy marketing layouts for technical docs.
- Recolor the final `Q` differently from the brand green.

## Quick Checklist

Before publishing:

- The favicon is visible and green.
- Every visible `GinkgoQ` has a green final `Q`.
- Light and dark modes both preserve contrast.
- Hover states do not create clipped rectangular bands.
- Mobile hero and cards remain readable.
- Code blocks, admonitions, and tables match the design tokens.
- External platform links use relevant icons and brand colors.
