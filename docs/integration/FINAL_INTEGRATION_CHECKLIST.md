# Final integration checklist

> **Historical integration record.** This checklist preserves the original
> merge evidence from `integration/final-mvp`. The current `main` product has
> since added R11/R12 hardening, deterministic evidence reconstruction,
> FinBERT/Integrated Gradients, filing chat, voice input, Docker, and Cloud Run.
> Current validation: **613 passed, 4 skipped, 0 failed, 4 warnings**. The four
> skips require `torch`, which was absent only from the audited local
> environment; it is declared in `requirements.txt` and installed in Docker.

Este checklist se ejecutará en una rama de integración dedicada. No autoriza
merges automáticos ni cambios en las ramas remotas de otros miembros.

Leyenda: `[x]` PASS (verificado en `integration/final-mvp`); `[x]` con nota
«caveat» = PASS WITH PROVIDER CAVEAT; `[ ]` = NOT EXECUTED / FUTURE WORK.
Verificación histórica de esta integración: hardening final, 2026-10-06.

## Dani — extraction and analysis

- [x] Contratos, pipeline y verifier implementados.
- [x] Contrato D8C serializable implementado y documentado.
- [x] Tests locales verdes antes de D9A.
- [ ] D8C/D9A revisados por el equipo.
- [ ] D8C/D9A committed y pushed cuando Dani lo autorice.
- [x] Tests verdes en la rama de integración final. — PASS en el checkpoint histórico; revalidación actual: 613 passed, 4 skipped, 0 failed

## Cristian — audio

- [x] `requirements.txt` de audio committed. — PASS: integrado en `requirements.txt`
- [ ] Rama de audio actualizada y pushed.
- [ ] Decidir y documentar el destino de `transcripccion_16abril.txt`.
- [x] Excluir cualquier transcript personal o no destinado al repositorio. — PASS: `transcripccion_*.txt` en `.gitignore`, ninguno versionado
- [x] Usar exclusivamente la API oficial en `src/api/`; no existe ni se requiere `scripts/dev_api.py`.
- [x] Reconciliar dependencias Groq/Kokoro/soundfile/python-dotenv. — PASS: declaradas; `pip check` limpio
- [x] Añadir tests automatizados de los contratos HTTP de audio. — PASS: cubiertos en `tests/test_api.py`
- [ ] Ejecutar smoke STT real con audio pequeño. — NOT EXECUTED en el checkpoint histórico; actualmente STT alimenta filing chat, no el pipeline de earnings calls
- [x] Ejecutar smoke TTS y comprobar WAV resultante. — PASS: `POST /api/v1/audio/summary` devolvió `audio/wav` válido (RIFF/WAVE, 24 kHz, mono) con Kokoro local
- [x] Integrar proveedor Groq TTS opcional para inglés sin sustituir Kokoro. — PASS contractual con mocks; coste/latencia y smoke real pendientes
- [ ] Comprobar el límite de 25 MB y el comportamiento del primer warm-up. — NOT EXECUTED: el modelo Kokoro ya estaba en caché; no se midió descarga en frío

## Marco — API, UI and visualization

- [ ] Publicar la rama de Marco.
- [x] Auditar `src/api/`, `src/visualization/` y `app/` antes del merge. — PASS: auditoría arquitectónica de solo lectura
- [x] Reconciliar respuestas API con `AnalysisHandoff`. — PASS: `response_model=AnalysisHandoff`; smoke demo y real con 200
- [x] Consumir métricas sin recalcular `change_pct` ni comparabilidad temporal. — PASS: auditoría de código
- [x] Mostrar `analysis_mode` y estado de verification. — PASS: comprobado con `streamlit.testing.AppTest` (no inspección visual)
- [x] Manejar `IntegrationError` sin exponer secretos o stack traces. — PASS: modo real con fecha inválida → 422 `INPUT_ERROR`; voz inválida → 422; sin traceback
- [x] Integrar positives, risks, outlook y executive summary. — PASS: secciones renderizadas según `AppTest`
- [x] Integrar controles de TTS usando únicamente `TTSInput.text`. — PASS: la UI envía `executive_summary` a `/api/v1/audio/summary`
- [x] Definir UX de fallback real/demo de forma explícita. — PASS: selector explícito, badge DEMO/REAL, sin fallback silencioso
- [x] Añadir tests de API/UI acordes con el contrato final. — PASS: `test_api.py`, `test_streamlit_app.py`, `test_visualization.py`

## Merge and dependency reconciliation

- [x] Crear rama de integración desde el commit base acordado. — PASS: `integration/final-mvp`
- [x] Merge de la rama de Dani. — PASS: 32fe3a2
- [x] Merge de `feature/audio_integration`. — PASS: eb84be3
- [x] Merge de la rama publicada de Marco. — PASS: ef553a2
- [x] Resolver `requirements.txt` sin perder pins o dependencias. — PASS
- [x] Verificar `.gitignore` para datos, `.env`, modelos, audio y transcripts. — PASS
- [x] Ejecutar `python -m pytest -q` con todos los módulos. — Estado actual: 613 passed, 4 skipped, 0 failed, 4 warnings
- [x] Ejecutar `python -m compileall src app` — PASS
- [ ] Ejecutar el lint oficial si Marco o la rama integrada lo configura. — NOT EXECUTED: no hay lint configurado en el repositorio
- [x] Ejecutar smoke SEC/XBRL con AAPL. — PASS: modo real AAPL 10-Q → SEC → pipeline → OpenRouter → 200, 7 métricas, verification válida
- [x] Ejecutar smoke del handoff API/UI. — PASS: demo y real vía FastAPI; UI vía `AppTest`
- [ ] Ejecutar smoke real STT/TTS en el entorno de presentación. — TTS histórico PASS; STT contractual cubierto por tests, smoke con Groq pendiente
- [ ] Ejecutar la secuencia completa de `docs/demo/DEMO_CHECKLIST.md`. — NOT EXECUTED: pendiente de ensayo manual con navegador
- [x] Revisar y actualizar el README con rutas/endpoints reales de Marco. — PASS
- [x] Confirmar que ningún FakeLLM aparece etiquetado como modo real. — PASS: demo → `provider=fixture`, `model=null`; real → `provider=openrouter`

## Delivery

- [x] Confirmar que no hay datos SEC, modelos, audio ni secretos versionados. — PASS
- [ ] Capturar versiones finales de Python y dependencias. — NOT EXECUTED
- [x] Verificar que el ejemplo de configuración no contiene claves reales. — PASS: placeholders en el README
- [ ] Validar manualmente visualizaciones y audio en el entorno de presentación. — NOT EXECUTED: no hubo inspección visual de la UI
- [ ] Documentar caveats externos conocidos sin ocultar fallos.
- [ ] Crear commit/tag final únicamente después de aprobación del equipo. — PENDIENTE
