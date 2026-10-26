# Pendientes de revisión — Memoria TFM

Depósito: 6 nov 2026. Marca con [x] lo que vayas cerrando.

## Capítulo 2 — Contexto

- [ ] Verificar en el PDF de MAGFiLO (Ahmadzadeh et al. 2024): κ = 0,66, proceso de revisión doble ciego, hasta 3 anotadores por imagen, 10.244 filamentos / 1.593 observaciones.
- [ ] Verificar la cita de GONG (Harvey et al. 1996, *Science*) y desde cuándo GONG toma imágenes H-alfa (no lo puse; quizá añadirlo).
- [ ] Revisar volumen/páginas: Zhu et al. 2019, Guo et al. 2022, Hao et al. 2013.
- [ ] Diercke et al. 2024 (A&A 686): añadir número de artículo.
- [ ] EdgeAttNet (Solomon et al. 2025): comprobar si ya está publicado o sigue solo en arXiv.
- [ ] Leer por encima las refs. físicas que no has leído: Martin 1998, Mackay et al. 2010, Parenti 2014, Schwenn 2006.
- [ ] Confirmar la afirmación "ningún trabajo previo usa topología ni PQ" (está como "hasta donde conocemos").
- [ ] Figura H-alfa completa con anotaciones (hueco comentado en el .tex).

## Capítulo 3 — Fundamentos de topología

- [ ] Repasar lo que no está en tus lecturas: complejos cúbicos, módulos de persistencia y descomposición en intervalos, estabilidad, dualidad de Alexander.
- [ ] Comprobar que las distancias bottleneck/Wasserstein y la estabilidad están en Edelsbrunner–Harer §VIII.2.
- [ ] Kaczynski et al. 2004: cambiar "cap. 2" por la definición concreta (no está en tu carpeta).
- [ ] La observación sobre 4/8-conectividad (construcciones V y T) es razonamiento propio, sin cita: revisarla.
- [ ] clDice (Shit et al. 2021) se cita pero no está en tu carpeta de bibliografía.
- [ ] Añadir figuras: construcción V vs T, diagrama de persistencia, grafo combinado de Topograph.
- [ ] Decidir si se recorta a ~10 págs. (candidatos: homología relativa, reducción de Smith, implementación eficiente).
- [ ] Dos líneas se salen ligeramente del margen.

## General

- [ ] Normativa del máster: límite de páginas y plantilla/portada obligatoria.
- [ ] Mantener Overleaf sincronizado: subir siempre `main.tex`, capítulos y `refs.bib` + "Recompile from scratch".
- [ ] Toda definición/proposición/teorema de un libro o paper lleva su referencia.

## Capítulo 4 — Planteamiento

- [ ] Las cifras del diagnóstico (recall 25 %/15 %/14 % vs 68 %, 203 perdidos por área, confianza 0,917 vs 0,933, 54 % espurios respaldados, PQ humano 0,34 vs modelo 0,38) son de la baseline del **notebook antiguo**: recalcular con la baseline del contrato v1 o indicarlo en el texto.
- [ ] Tabla de oráculos: rellenar con `oraculos.csv` (no está en tu carpeta).
- [ ] Comprobar que las definiciones de la taxonomía (fragmentado, fusionado, mal delimitado) coinciden con `classify()` del notebook.
- [ ] Figura con casos de *Imagenes de problemas* (copiar a `figuras/` y descomentar).
- [ ] E2 (persistencia): definir bien τ y δ al implementarlo; si no llega, quitar E2 o pasarlo a trabajo futuro.
- [ ] Ajustar el párrafo "Alcance" según lo que finalmente tenga resultados (Topograph, E4).

## Capítulos por escribir

- [ ] Cap. 5 — Metodología (baseline, contrato/split A/B, pérdidas, post-proceso)
- [ ] Cap. 6 — Experimentos y resultados
- [ ] Cap. 1 — Introducción y objetivos
- [ ] Cap. 7 — Conclusiones (incluir trabajo futuro: lo que no llegue de E2–E4)
- [ ] Resumen / abstract
- [ ] Verificar las referencias Dice (1945) y Sørensen (1948) añadidas en refs.bib: escritas de memoria, revisar volumen y páginas.
- [ ] Capítulo 3: sección «Dice y clDice» nueva. Comprobar las definiciones de clDice y del esqueleto blando frente a Shit et al. (2021), Sec. 3.
