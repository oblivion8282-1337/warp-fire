# Echtzeit-Feuer-Rendering in Blender — Recherche

## Stand der Forschung (2025/2026)

### Aktuelle Papers & Technologien

#### 1. "Fire Simulation in CG: A Survey" (Guo Yixuan, 2025)
- **Quelle**: Scientific Journal of Intelligent Systems Research, Vol. 7, Issue 12
- **Link**: https://bcpublication.org/index.php/SJISR/article/view/9075
- Umfassender Review von Feuer-Simulationsmethoden (2016–2025)
- Kategorien:
  - **Physik-basiert**: Navier-Stokes-Gleichungen, Vortex Methods, Level-Set
  - **Data-driven**: CNN-basierte Feature-Descriptors, Neural Networks für Ausbreitungsparameter
  - **Visual-Effects**: Noise-basiert, prozedural (für Games/Echtzeit)
  - **Hybrid**: Kombination physik-basiert + neural
- Vergleich der Vor-/Nachteile jeder Methode

#### 2. NeuralVDB (NVIDIA)
- **Link**: https://developer.nvidia.com/rendering-technologies/neuralvdb
- AI-basierte Kompression für OpenVDB-Volumendaten (Feuer, Rauch, Wolken)
- Bis zu **100x kleinerer Memory-Footprint**
- Kompatibel mit bestehenden VDB-Pipelines
- Ermöglicht Echtzeit-Interaktion mit komplexen volumetrischen Datasets

#### 3. ZibraVDB (2025, kostenlos)
- **Link**: https://www.zibra.ai
- Komprimiert offline VDB-Simulationen bis zu 100x
- Echtzeit-Rendering in **Unreal Engine 5**
- Seit Januar 2025 kostenlos verfügbar
- Relevant für UE5-Workflows, nicht direkt für Blender

#### 4. SIGGRAPH 2025 — Verwandte Arbeiten
- **"Don't Splat your Gaussians"** (Meta Reality Labs / USI Lugano)
  - Volumetric Ray-Traced Primitives für emissive/streuende Medien
  - Anwendbar auf Feuer, Rauch, Nebel
- **Painterly Visualization of Smoke and Fire**
  - Stroke-basierte stilisierte Darstellung
  - Stroke/Basis Fields für Lighting-, View- und Geometry-abhängige Infos
- **NVIDIA Hybrid Subsurface Scattering** mit ReSTIR Sampling
- **Papers-Index**: https://www.realtimerendering.com/kesen/sig2025.html

#### 5. Klassiker (weiterhin referenziert)
- **Horvath & Geiger: "Directable, High-Resolution Simulation of Fire on the GPU"**
  - GPU-beschleunigte Feuer-Simulation
  - Wird in aktuellen Papers immer noch zitiert
  - https://history.siggraph.org/learning/directable-high-resolution-simulation-of-fire-on-the-gpu-by-horvath-and-geiger/

---

## Ansätze für ein Blender-Plugin (Echtzeit)

### Option 1: Prozedurale GLSL-Shader (schnellste, rein visuell)
- Blenders `gpu`-Modul für custom Fragment-Shaders im Viewport
- Ray Marching durch ein 3D-Noise-Volume (fBm, Curl Noise, Worley)
- Kein Fluid-Solver nötig — rein visuell, aber extrem schnell
- Klassischer Game-Engine-Ansatz
- **Performance**: Tausende FPS auf RTX 4090
- **Aufwand**: ~100-200 Zeilen Shader-Code + Python-Wrapper
- **Limitierung**: Kein echtes physikalisches Verhalten, keine Interaktion mit Geometrie

### Option 2: NVIDIA Warp (bester Kompromiss)
- **Link**: https://github.com/NVIDIA/warp
- Python-Framework, kompiliert direkt zu **CUDA Kernels**
- Lässt sich aus Blender-Python aufrufen (`pip install warp-lang`)
- Echten Fluid-Solver auf GPU: Semi-Lagrangian Advection, Buoyancy, Vorticity Confinement
- Ergebnis als Volume/Mesh zurück in den Viewport speisen
- **Performance**: Echtzeit für moderate Grid-Größen (128³–256³)
- **Aufwand**: ~200-300 Zeilen Python für minimalen Smoke/Fire-Solver
- **Limitierung**: CUDA-only (nur NVIDIA GPUs)

### Option 3: Taichi Lang
- **Link**: https://github.com/taichi-dev/taichi
- Python-DSL, kompiliert zu CUDA/Vulkan/Metal
- Fertige Fluid-Simulation-Beispiele vorhanden
- Multi-Backend (nicht nur NVIDIA)
- **Performance**: Vergleichbar mit Warp
- **Aufwand**: Ähnlich wie Warp, etwas mehr Boilerplate
- **Vorteil**: Läuft auch auf AMD GPUs (Vulkan-Backend)

### Option 4: Geometry Nodes + EEVEE Shader (Blender-nativ)
- Kein externer Code, alles in Blender
- Partikel über Geometry Nodes emittieren, Fire-Shader in EEVEE
- **Performance**: Gut für einfache Effekte
- **Aufwand**: Minimal
- **Limitierung**: Keine echte Simulation, begrenzte Kontrolle

---

## Empfehlung

| Ziel | Empfohlener Ansatz |
|------|-------------------|
| Nur Visuals, max. Performance | Option 1 — GLSL Ray Marching |
| Echte Simulation + Echtzeit | Option 2 — NVIDIA Warp |
| Cross-GPU-Kompatibilität | Option 3 — Taichi Lang |
| Schneller Prototyp, kein Code | Option 4 — Geometry Nodes |

Für RTX 4090 + Blender ist **NVIDIA Warp** der Sweet Spot:
Python-Code wird zu CUDA kompiliert, volle GPU-Power, nahtlose Blender-Integration.

---

## Technische Bausteine für die Implementierung

### Minimaler Fire-Solver (Warp/Taichi)
1. **3D-Grid** (z.B. 128³) für Temperatur, Dichte, Velocity
2. **Advection**: Semi-Lagrangian (stabil, schnell)
3. **Buoyancy**: Temperatur → Aufwärtskraft
4. **Vorticity Confinement**: Wirbel erhalten für Details
5. **Combustion**: Fuel → Temperature + Smoke Density
6. **Dissipation**: Temperatur/Dichte über Zeit abbauen
7. **Rendering**: Temperature → Color (Blackbody Curve), Density → Alpha

### Blender-Integration
- `bpy.types.Operator` für Start/Stop
- `gpu`-Modul oder `bpy.types.SpaceView3D.draw_handler_add` für Viewport-Overlay
- Timer (`bpy.app.timers`) für Frame-Updates
- Optional: OpenVDB-Export für Cycles-Rendering

## Weiterführende Links
- SIGGRAPH 2025 Papers: https://www.realtimerendering.com/kesen/sig2025.html
- Advances in Real-Time Rendering 2025: https://www.advances.realtimerendering.com/s2025/index.html
- NVIDIA Real-Time Rendering Research: https://research.nvidia.com/labs/rtr/
- SIGGRAPH 2026 Call for Papers: https://egsr2026.inria.fr/call-for-papers/
