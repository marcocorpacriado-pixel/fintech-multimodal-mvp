# Pitch visual — guía de pantallas

Guía de la interfaz para analistas y bloques listos para enlazar las capturas desde `README.md` y `BUSINESS_CASE.md`. El sistema de diseño completo está en [`DESIGN.md`](../DESIGN.md).

## Cómo capturar

1. Arrancar API y UI desde la raíz (`uvicorn src.api.main:app --port 8000` y `streamlit run app/streamlit_app.py`).
2. Ventana a 1440 px de ancho, zoom 100 %, chat flotante cerrado.
3. Usar el mismo ticker y filing en las cuatro capturas (en modo DEMO, el badge `DEMO | SYNTHETIC` debe verse).
4. Guardar en PNG con los nombres exactos de abajo dentro de `docs/assets/`.

| Archivo | Pestaña UI | Estado |
|---|---|---|
| `tab1_executive_summary_audio.png` | Overview | pendiente de captura |
| `tab2_financial_metrics_yoy.png` | Financials | pendiente de captura |
| `tab3_drivers_xai_outlook.png` | Narrative | pendiente de captura |
| `tab4_compliance_verification.png` | Sources | pendiente de captura |

Para la captura 1, el reproductor de audio está al final de la pestaña Narrative: capturar Overview y, si se quiere mostrar el audio en la misma imagen, recortar el reproductor de Narrative.

## Pantallas

### 1. Resumen ejecutivo y audio

Snapshot del filing en una sola vista: KPIs clave, periodo reportado, fecha de presentación y estado LIVE / DEMO en la cabecera fija. El resumen se puede escuchar vía TTS.
**Propuesta de valor:** menos fricción cognitiva. El analista sabe en segundos si el filing merece lectura profunda, sin abrir 100+ páginas.

### 2. Métricas YoY

Siete métricas canónicas extraídas de XBRL por un pipeline determinista, comparadas con el periodo anterior en barras duales (Actual vs Anterior) y cambio porcentual.
**Propuesta de valor:** trazabilidad contable. Cada cifra viene del XBRL oficial y la UI no recalcula nada; lo que se ve es lo que la SEC publicó.

### 3. Drivers y outlook XAI

Positivos y riesgos con la sección SEC de origen, más el outlook de gestión clasificado por FinBERT. La frase de evidencia muestra un heatmap de Integrated Gradients: cuanto más intensa la pill, más pesa la palabra; al pasar el cursor o enfocar con teclado aparece su impacto.
**Propuesta de valor:** explicabilidad algorítmica. El modelo no da un veredicto opaco; enseña qué palabras del filing lo han llevado a esa polaridad.

### 4. Compliance y verificación

Evidencia citada literal, resultado del verifier determinista (cada afirmación contrastada con el texto fuente) y detalles técnicos del run.
**Propuesta de valor:** confianza auditable. Cualquier conclusión se puede rastrear hasta el fragmento exacto del filing.

## Bloques para copiar

Desde `README.md` (raíz del repo):

```markdown
### Resumen ejecutivo y audio
![Resumen ejecutivo y audio](docs/assets/tab1_executive_summary_audio.png)

### Métricas YoY
![Métricas financieras YoY](docs/assets/tab2_financial_metrics_yoy.png)

### Drivers y outlook XAI
![Drivers y outlook con heatmap XAI de FinBERT](docs/assets/tab3_drivers_xai_outlook.png)

### Compliance y verificación
![Evidencia y verificación determinista](docs/assets/tab4_compliance_verification.png)
```

Desde un documento dentro de `docs/` (p. ej. `docs/BUSINESS_CASE.md`), las rutas pierden el prefijo `docs/`:

```markdown
![Resumen ejecutivo y audio](assets/tab1_executive_summary_audio.png)
![Métricas financieras YoY](assets/tab2_financial_metrics_yoy.png)
![Drivers y outlook con heatmap XAI de FinBERT](assets/tab3_drivers_xai_outlook.png)
![Evidencia y verificación determinista](assets/tab4_compliance_verification.png)
```

Si `BUSINESS_CASE.md` vive en la raíz, usar el primer bloque.
