# Final integration checklist

Este checklist se ejecutará en una rama de integración dedicada. No autoriza
merges automáticos ni cambios en las ramas remotas de otros miembros.

## Dani — extraction and analysis

- [x] Contratos, pipeline y verifier implementados.
- [x] Contrato D8C serializable implementado y documentado.
- [x] Tests locales verdes antes de D9A.
- [ ] D8C/D9A revisados por el equipo.
- [ ] D8C/D9A committed y pushed cuando Dani lo autorice.
- [ ] Tests verdes en la rama de integración final.

## Cristian — audio

- [ ] `requirements.txt` de audio committed.
- [ ] Rama de audio actualizada y pushed.
- [ ] Decidir y documentar el destino de `transcripccion_16abril.txt`.
- [ ] Excluir cualquier transcript personal o no destinado al repositorio.
- [ ] Decidir si `scripts/dev_api.py` se publica o permanece fuera de integración.
- [ ] Reconciliar dependencias Groq/Kokoro/soundfile/python-dotenv.
- [ ] Añadir o acordar tests automatizados mínimos de contratos de audio.
- [ ] Ejecutar smoke STT con audio pequeño.
- [ ] Ejecutar smoke TTS y comprobar WAV resultante.
- [ ] Comprobar el límite de 25 MB y el comportamiento del primer warm-up.

## Marco — API, UI and visualization

- [ ] Publicar la rama de Marco.
- [ ] Auditar `src/api/`, `src/visualization/` y `app/` antes del merge.
- [ ] Reconciliar respuestas API con `AnalysisHandoff`.
- [ ] Consumir métricas sin recalcular `change_pct` ni comparabilidad temporal.
- [ ] Mostrar `analysis_mode` y estado de verification.
- [ ] Manejar `IntegrationError` sin exponer secretos o stack traces.
- [ ] Integrar positives, risks, outlook y executive summary.
- [ ] Integrar controles de TTS usando únicamente `TTSInput.text`.
- [ ] Definir UX de fallback real/demo de forma explícita.
- [ ] Añadir tests de API/UI acordes con el contrato final.

## Merge and dependency reconciliation

- [ ] Crear rama de integración desde el commit base acordado.
- [ ] Merge de la rama de Dani.
- [ ] Merge de `feature/audio_integration`.
- [ ] Merge de la rama publicada de Marco.
- [ ] Resolver `requirements.txt` sin perder pins o dependencias.
- [ ] Verificar `.gitignore` para datos, `.env`, modelos, audio y transcripts.
- [ ] Ejecutar `python -m pytest -q` con todos los módulos.
- [ ] Ejecutar `python -m compileall src`.
- [ ] Ejecutar el lint oficial si Marco o la rama integrada lo configura.
- [ ] Ejecutar smoke SEC/XBRL con AAPL.
- [ ] Ejecutar smoke del handoff API/UI.
- [ ] Ejecutar smoke STT/TTS.
- [ ] Ejecutar la secuencia completa de `docs/demo/DEMO_CHECKLIST.md`.
- [ ] Revisar y actualizar el README con rutas/endpoints reales de Marco.
- [ ] Confirmar que ningún FakeLLM aparece etiquetado como modo real.

## Delivery

- [ ] Confirmar que no hay datos SEC, modelos, audio ni secretos versionados.
- [ ] Capturar versiones finales de Python y dependencias.
- [ ] Verificar que el ejemplo de configuración no contiene claves reales.
- [ ] Validar manualmente visualizaciones y audio en el entorno de presentación.
- [ ] Documentar caveats externos conocidos sin ocultar fallos.
- [ ] Crear commit/tag final únicamente después de aprobación del equipo.
