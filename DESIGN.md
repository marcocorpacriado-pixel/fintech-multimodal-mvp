# Fintech Multimodal Design System (Dark Slate Terminal)

## Mission

Proporcionar una interfaz analítica institucional de alto contraste, sobria y libre de distracciones para la inspección y síntesis multimodal de filings SEC (10-K / 10-Q), garantizando máxima legibilidad, navegación por pestañas y trazabilidad de datos.

## Brand & Context

- **Product:** Fintech Multimodal MVP
- **Audience:** Analistas financieros, gestores de carteras y auditores cuantitativos.
- **Surface:** Dashboard interactivo (Streamlit + Plotly).

## Style Foundations

- **Visual style:** Financial terminal, dense data density, dark slate, crisp contrast.
- **Color Palette:**

| Token | Valor | Uso |
|---|---|---|
| `background-base` | `#0B0F19` | Slate ultra-oscuro (fondo de la app) |
| `surface-card` | `#1E293B` | Slate medio para tarjetas de métricas, sidebar |
| `border-subtle` | `#334155` | Bordes, divisores y gridlines |
| `text-primary` | `#F8FAFC` | Blanco frío de alto contraste |
| `text-muted` | `#94A3B8` | Texto secundario y metadatos |
| `accent-blue` | `#38BDF8` / `#0284C7` | Acciones primarias, indicadores, serie "Actual" |
| `positive-growth` | `#10B981` | Esmeralda para drivers positivos |
| `warning-risk` | `#F59E0B` | Ámbar para riesgos y advertencias |
| `neutral-previous` | `#64748B` | Periodo de referencia en comparativas (serie "Anterior") |

- **Typography:** Inter, Segoe UI, sans-serif. Valores clave 24px bold; etiquetas 12px uppercase semibold.
- **Radii:** 8px para tarjetas y contenedores.

### Implementación de los tokens

| Capa | Dónde | Qué cubre |
|---|---|---|
| Tema nativo | `.streamlit/config.toml` | Colores base, fuente Inter, radio 8px, bordes, tamaño de valor KPI, colores de badges |
| Componentes nativos | `app/streamlit_app.py` | Jerarquía, tarjetas, pestañas y detalle expandible |
| CSS inyectado | `app/streamlit_app.py` (`TERMINAL_CSS`) | Padding vertical, etiquetas KPI uppercase, cabecera sticky |
| Plotly | `src/visualization/financial_charts.py` | Comparativas y cambios porcentuales sin recalcular métricas |

Streamlit lee `.streamlit/config.toml` desde el directorio de trabajo: arrancar siempre desde la raíz (`streamlit run app/streamlit_app.py`).

## Component Layout Rules (Streamlit Native)

- **Header:** Sticky. Ticker, periodo reportado, filing date, tipo de filing y estado LIVE / DEMO.
- **Sidebar (real):** Ticker desde lista cerrada (`TICKERS`); los filings se descubren en vivo vía `GET /api/v1/filings/{ticker}`.
- **Tabs (`st.tabs`):**
  1. **Overview:** snapshot ejecutivo y mensajes clave.
  2. **Financials:** siete métricas canónicas, gráficos y tabla detallada.
  3. **Narrative:** positivos, riesgos, outlook, resumen ejecutivo y TTS.
  4. **Sources:** evidencia, verificación y detalles técnicos.

## Rules: Do

- Utilizar `st.tabs` para dividir la carga cognitiva en lugar de una página vertical infinita.
- Respetar los valores brutos de la API: `change_pct` ya viene en puntos porcentuales; `None` se renderiza "N/D".
- Plotly con fondo transparente (`paper_bgcolor` y `plot_bgcolor` en `"rgba(0,0,0,0)"`) y gridlines sutiles (`#334155`).
- Escapar todo texto procedente de filings (`md_escape`) antes de renderizarlo como markdown.

## Rules: Don't

- Prohibido recalcular métricas o multiplicar por 100.
- Prohibido importar módulos de backend o IA en la capa UI (desacoplamiento total vía HTTP contra `localhost:8000`).
- No usar color como único portador de significado: cada badge y serie lleva también su etiqueta de texto.
