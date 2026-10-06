# Demo checklist

Checklist operativo para una presentación reproducible. No presupone botones,
endpoints ni pantallas concretas de la futura implementación de Marco.

## Pre-demo

- [ ] Activar el entorno virtual correcto.
- [ ] Instalar las dependencias reconciliadas de la rama de integración.
- [ ] Confirmar que `.env` y secretos no están versionados.
- [ ] Verificar `EDGAR_IDENTITY` para acceso SEC real.
- [ ] Verificar `OPENROUTER_API_KEY` y `OPENROUTER_MODEL` si se usará modo real.
- [ ] Verificar `GROQ_API_KEY` si se demostrará STT real.
- [ ] Confirmar explícitamente `analysis_mode="real"` o `"demo"`.
- [ ] Preparar la fixture validada si se usará modo demo.
- [ ] Confirmar disponibilidad del filing y XBRL seleccionados.
- [ ] Ejecutar `python -m pytest -q` o, como mínimo, el smoke acordado.
- [ ] Hacer warm-up de Kokoro y confirmar la voz seleccionada.
- [ ] Preparar un WAV de respaldo generado desde el mismo executive summary.
- [ ] Si habrá STT, comprobar formato y tamaño del audio (máximo práctico 25 MB).

## Secuencia de demo

1. Seleccionar empresa y filing objetivo.
2. Mostrar las métricas canónicas current/previous y su tipo de comparación.
3. Mostrar desarrollos positivos con su evidencia.
4. Mostrar riesgos con su evidencia.
5. Mostrar sentiment y resumen del management outlook.
6. Mostrar el executive summary.
7. Mostrar `verification.valid` y cualquier warning no bloqueante.
8. Generar o reproducir el TTS del executive summary.
9. Opcionalmente demostrar STT con un audio corto y previamente validado.

## Validaciones durante la presentación

- [ ] La UI muestra el `analysis_mode` real.
- [ ] Las métricas coinciden con el `AnalysisHandoff`; la UI no las recalcula.
- [ ] Evidence, section y source ID permanecen asociados.
- [ ] Un resultado con verification errors no se presenta como válido.
- [ ] El texto enviado a TTS coincide con `executive_summary`.

## Fallbacks explícitos

- Si OpenRouter falla o excede la latencia disponible, detener el flujo real y
  reiniciar la demo con la fixture validada y `analysis_mode="demo"` visible.
- Si Groq/STT falla, usar un transcript preparado e identificarlo como tal.
- Si Kokoro tarda en iniciar, utilizar el modelo pre-warmed o el WAV preparado.
- Si SEC no está disponible, usar los inputs locales previamente materializados
  y explicar que no se está haciendo una descarga en vivo.
- No ocultar ningún fallback ni presentar resultados demo como inferencia real.

## Post-demo

- [ ] No conservar secretos, headers ni prompts en logs compartidos.
- [ ] Eliminar audios temporales que no deban persistir.
- [ ] Registrar cualquier fallo como provider, configuración, grounding,
  verification, SEC/XBRL, audio o presentación antes de proponer cambios.
