# Fintech Multimodal Design System (Dark Slate Terminal)

## Mission

Proporcionar una interfaz analítica institucional de alto contraste, sobria y libre de distracciones para la inspección y síntesis multimodal de filings SEC (10-K / 10-Q), garantizando máxima legibilidad, navegación por pestañas y trazabilidad de datos.

## Brand & Context

- **Product:** Fintech Multimodal MVP (Financial Intelligence Copilot)
- **Audience:** Analistas financieros, gestores de carteras y auditores cuantitativos.
- **Surface:** Dashboard interactivo (Dash + Plotly) que consume `AnalysisHandoff` vía HTTP.
- **Visual style:** Financial terminal, densidad de datos alta, dark slate, contraste nítido.

## 1. Foundations & Tokens

### Color

| Token | Valor | Uso | Contraste vs `background-base` / `surface-card` |
|---|---|---|---|
| `background-base` | `#0B0F19` | Fondo de la app, cabecera sticky, tooltips | — |
| `surface-card` | `#1E293B` | Tarjetas, sidebar, bloque de evidencia XAI | — |
| `border-subtle` | `#334155` | Bordes, divisores, gridlines | decorativo |
| `text-primary` | `#F8FAFC` | Texto principal y valores KPI | 18.30:1 / 13.98:1 |
| `text-muted` | `#94A3B8` | Texto secundario, captions, etiquetas KPI, metadatos, leyenda XAI | 7.47:1 / 5.71:1 |
| `accent-blue` | `#38BDF8` | Links, foco, serie "Actual", polaridad neutral | 8.94:1 / 6.83:1 |
| `accent-primary` | `#0284C7` | `--primary` en `app/assets/style.css` (botones primarios, relleno) | no se usa como texto sobre fondo oscuro |
| `sentiment-positive` (emerald) | `#10B981` | Drivers positivos, polaridad positiva | 7.55:1 / 5.77:1 |
| `sentiment-negative` (crimson) | `#EF4444` | Riesgos, polaridad negativa (solo XAI) | 5.09:1 / 3.89:1 |
| `sentiment-unscored` (slate) | `#94A3B8` | Polaridad sin puntuación / desconocida | 7.47:1 / 5.71:1 |
| `warning-amber` | `#F59E0B` | Badges DEMO / advertencias | 8.92:1 / 6.81:1 |
| `neutral-previous` | `#64748B` | Serie "Anterior" en Plotly (**solo gráfico, nunca texto**) | 4.02:1 (≥ 3:1 no-texto) |

`#64748B` no cumple 4.5:1 para texto y queda reservado a marcas gráficas. Cualquier texto secundario usa `text-muted` (`#94A3B8`).

### Tipografía y forma

- **Familia:** Inter (400/600/700 vía Google Fonts), fallback Segoe UI, sans-serif.
- **Escala:** valores KPI 24px bold; etiquetas 12px uppercase semibold, `letter-spacing: 0.04em`; cita de evidencia XAI 15px, `line-height: 2.1`.
- **Radios:** 8px para tarjetas y contenedores (`baseRadius`); 4px en pills y tooltips; 999px en badges de polaridad y en el botón flotante de chat.

### Implementación de los tokens

| Capa | Dónde | Qué cubre |
|---|---|---|
| Tokens y CSS | `app/assets/style.css` (variables `:root`) | Colores base, `--muted` (= `text-muted`), fuente Inter (Google Fonts), radio 8px, etiquetas y valores KPI, badges, cabecera sticky, tooltip instantáneo `.xai-pill`, foco, chat flotante, tema de `dcc.Dropdown` y `dcc.Slider` |
| Vistas | `app/components.py` | Componentes Dash puros (sin callbacks ni HTTP) que construyen cada sección a partir del `AnalysisHandoff` |
| Tema Plotly | `app/components.py` (`themed`) | `plotly_dark`, fondo transparente y fuente Inter sobre las figuras de `src/visualization` |
| HTML de presentación | `src/visualization/presentation.py` | Heatmap XAI, badge de polaridad, leyenda de atribución |
| Plotly | `src/visualization/financial_charts.py` | Comparativas y cambios porcentuales sin recalcular métricas |

Dash sirve automáticamente `app/assets/` (CSS y JS). Arranque local desde la raíz: `python -m app.dash_app`; producción: `gunicorn app.dash_app:server`.

## 2. Component Specifications

### Layout de 5 pestañas (`dcc.Tabs`, `id="result-tabs"`)

Las pestañas se resuelven en el cliente: cambiar de pestaña o cerrar el chat no vuelve a pedir nada al servidor. Las etiquetas en la UI están en inglés; la columna "Nombre de pitch" es la que se usa en presentaciones.

| # | Etiqueta UI | Nombre de pitch | Contenido |
|---|---|---|---|
| 1 | Overview | Resumen ejecutivo | Snapshot ejecutivo: KPIs principales y mensajes clave |
| 2 | Financials | Métricas YoY & Plotly | Siete métricas canónicas, badges de comparación, gráficos Plotly y tabla detallada |
| 3 | Narrative | Drivers & XAI Outlook (+ Audio) | Positivos y riesgos con su fuente, outlook FinBERT con heatmap XAI, resumen ejecutivo y reproductor TTS |
| 4 | Sources | Compliance & Verificación | Evidencia citada, resultado del verifier determinista y detalles técnicos |
| 5 | Performance | Latencia y coste | Latencia y coste medidos por inferencia en la sesión y proyección de viabilidad mensual |

- **Header sticky:** ticker, periodo reportado, filing date, tipo de filing y badge LIVE / DEMO (texto + color).
- **Sidebar (real):** ticker desde lista cerrada (`TICKERS`); filings descubiertos vía `GET /api/v1/filings/{ticker}`.
- **Chat flotante:** botón fijo abajo a la derecha; panel translúcido (`rgba(11,15,25,0.8)`) no modal para seguir viendo el análisis.

### Componente XAI (heatmap FinBERT)

Generado por `highlight_tokens_html` / `outlook_xai_html` y renderizado con `dcc.Markdown(dangerously_allow_html=True)` tras colapsar espacios en blanco. Todo el texto del filing se escapa con `html.escape`; solo los spans son markup.

- **Badge de polaridad:** pill 999px, fondo `rgba(<sentimiento>, 0.10)`, borde `rgba(<sentimiento>, 0.6)`, texto en el color de sentimiento, texto `● POLARITY: <LABEL>` (el color nunca va solo).
- **Pills `.xai-pill`:** una por palabra con atribución > 0.
  - Fondo `rgba(rgb, 0.15 + score × 0.35)` → alfa 0.15–0.50.
  - Borde `rgba(rgb, 0.3 + score × 0.4)` → alfa 0.30–0.70.
  - Texto pálido del sentimiento (`#E6FFFA` emerald, `#FFF5F5` crimson, `#F0F9FF` neutral, `#F8FAFC` slate).
  - El tope de alfa 0.50 es deliberado: con 0.60 la pill neutral bajaba a 4.02:1.
- **Interacción:** `cursor: help`; `tabindex="0"` para foco por teclado; `data-tooltip="Impact: NN.N%"` alimenta un tooltip CSS instantáneo (`::after` en `:hover` y `:focus-visible`, fondo `#0B0F19`, borde `#334155`); `title` como fallback nativo accesible.
- **Foco:** `outline: 2px solid #38BDF8; outline-offset: 1px`.
- **Leyenda:** "Low impact" → gradiente 84×8px `rgba(rgb, 0.15)` → `rgba(rgb, 0.5)` → "High impact · Integrated Gradients attribution", en `text-muted` 12px.

### Gráficos Plotly

- Fondos transparentes: `paper_bgcolor` y `plot_bgcolor` en `"rgba(0,0,0,0)"`; gridlines sutiles.
- Barras duales: **Actual `#38BDF8`** vs **Anterior `#64748B`**, con leyenda textual.
- Cambio porcentual en `#607d8b`: neutral a propósito, la dirección no es sentimiento de inversión.
- Un panel por unidad (USD / per-share); tipografía compacta heredada de Inter.
- Valores tal como llegan de la API: `change_pct` ya viene en puntos porcentuales; `None` → "N/A".

## 3. WCAG 2.2 AA Compliance

Ratios calculados con la fórmula de luminancia relativa de WCAG. Mínimos: 4.5:1 para texto normal y 3:1 para componentes no textuales (1.4.11).

| Elemento | Primer plano / fondo | Ratio | Criterio |
|---|---|---|---|
| Texto principal | `#F8FAFC` / `#1E293B` | 13.98:1 | ✅ 1.4.3 |
| Texto secundario, captions, etiquetas KPI | `#94A3B8` / `#1E293B` | 5.71:1 | ✅ 1.4.3 |
| Texto secundario sobre fondo base | `#94A3B8` / `#0B0F19` | 7.47:1 | ✅ 1.4.3 |
| Pill XAI, peor caso (neutral, score 1.0) | `#F0F9FF` / sky 0.50 sobre `#1E293B` | 4.90:1 | ✅ 1.4.3 |
| Pill XAI, crimson score 1.0 | `#FFF5F5` / red 0.50 sobre `#1E293B` | 7.51:1 | ✅ 1.4.3 |
| Badge de polaridad, peor caso (negativa) | `#EF4444` / red 0.10 sobre `#0B0F19` | 4.71:1 | ✅ 1.4.3 |
| Serie "Anterior" | `#64748B` / `#0B0F19` | 4.02:1 | ✅ 1.4.11 |
| Serie "Actual" | `#38BDF8` / `#0B0F19` | 8.94:1 | ✅ 1.4.11 |

- **Teclado:** pills XAI enfocables (`tabindex="0"`), tooltip visible también en `:focus-visible`; controles nativos y de Dash navegables con Tab.
- **Foco visible (2.4.7 / 2.4.11):** outline 2px `#38BDF8` (8.94:1 sobre base).
- **Uso del color (1.4.1):** cada badge, serie y polaridad lleva etiqueta textual.
- **Tooltip (1.4.13):** aparece en hover y foco, no tapa el disparador y desaparece al salir. Limitación conocida: al ser CSS puro no se descarta con Esc; el mismo dato está en `title`.

## Rules: Do

- Utilizar pestañas (`dcc.Tabs`) para dividir la carga cognitiva en lugar de una página vertical infinita.
- Respetar los valores brutos de la API: `change_pct` ya viene en puntos porcentuales; `None` se renderiza "N/A".
- Escapar todo texto procedente de filings (como hijos de componentes Dash, que React escapa, y `html.escape` en el HTML del heatmap).
- Verificar el contraste de cualquier color nuevo contra `#0B0F19` y `#1E293B` antes de usarlo como texto.

## Rules: Don't

- Prohibido recalcular métricas o multiplicar por 100.
- Prohibido importar módulos de backend o IA en la capa UI (desacoplamiento total vía HTTP contra `localhost:8000`).
- No usar color como único portador de significado.
- No usar `#64748B` (ni grises más oscuros) para texto.
