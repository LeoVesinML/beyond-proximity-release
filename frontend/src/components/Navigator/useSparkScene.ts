/**
 * useSparkScene — loads a .ply Gaussian Splatting scene via @sparkjsdev/spark
 * into a THREE.js Scene and manages the animation loop.
 *
 * Both THREE.js and Spark.js are loaded from CDN so they share the same module
 * instance and the same ShaderChunk registry (Spark registers custom GLSL chunks
 * like `splatDefines` at startup; the renderer must be from the same THREE build).
 */
import type * as THREETypes from 'three'
import { useEffect, useRef, useState } from 'react'
import { CAMERA_NEAR, CAMERA_FAR, type SparkDepthSource } from './DepthCapture'

// CDN URLs — must match exactly so the browser module cache gives one shared instance.
// CDN Spark.js imports "three" via the importmap in index.html, which points to the
// same THREE_CDN URL, ensuring both renderer and Spark share ShaderChunk state.
const THREE_CDN = 'https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js'
const SPARK_CDN = 'https://sparkjs.dev/releases/spark/0.1.10/spark.module.min.js'

interface SparkSceneResult {
  renderer: THREETypes.WebGLRenderer | null
  scene: THREETypes.Scene | null
  camera: THREETypes.PerspectiveCamera | null
  depthSource: SparkDepthSource | null
  ready: boolean
  error: string | null
}

interface UseSparkSceneOptions {
  canvasRef: React.RefObject<HTMLCanvasElement>
  plyUrl: string | null
  fov?: number
}

export function useSparkScene({
  canvasRef,
  plyUrl,
  fov = 60,
}: UseSparkSceneOptions): SparkSceneResult {
  const rendererRef = useRef<THREETypes.WebGLRenderer | null>(null)
  const sceneRef    = useRef<THREETypes.Scene | null>(null)
  const cameraRef   = useRef<THREETypes.PerspectiveCamera | null>(null)
  const sparkRef    = useRef<unknown>(null)
  const splatRef    = useRef<unknown>(null)
  const depthSourceRef = useRef<SparkDepthSource | null>(null)
  const rafRef      = useRef<number>(0)

  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    setReady(false)
    setError(null)

    const w = canvas.clientWidth  || canvas.offsetWidth  || 800
    const h = canvas.clientHeight || canvas.offsetHeight || 600
    let disposed = false

    // Animation loop starts immediately; renders once refs are populated.
    const animate = () => {
      rafRef.current = requestAnimationFrame(animate)
      const r = rendererRef.current
      const s = sceneRef.current
      const c = cameraRef.current
      if (r && s && c) r.render(s, c)
    }
    animate()

    const onResize = () => {
      const nw = canvas.clientWidth  || 800
      const nh = canvas.clientHeight || 600
      rendererRef.current?.setSize(nw, nh)
      const c = cameraRef.current
      if (c) { c.aspect = nw / nh; c.updateProjectionMatrix() }
    }
    window.addEventListener('resize', onResize)

    ;(async () => {
      try {
        // Load CDN THREE.js — same URL the importmap maps "three" to, so CDN
        // Spark.js (which does `import "three"` via importmap) shares this instance.
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        const THREE = await import(/* @vite-ignore */ THREE_CDN) as unknown as typeof THREETypes

        if (disposed) return

        // Create renderer / scene / camera from CDN THREE so they share ShaderChunk
        // with Spark.js (which registers splatDefines and other custom chunks).
        // preserveDrawingBuffer: true is required so canvas.toDataURL() in useViewCapture
        // captures a non-empty image — without it, WebGL clears the buffer after compositing.
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true, preserveDrawingBuffer: true })
        renderer.setPixelRatio(window.devicePixelRatio)
        renderer.setSize(w, h)
        rendererRef.current = renderer

        const scene = new THREE.Scene()
        scene.background = new THREE.Color(0x0d1117)
        sceneRef.current = scene

        const camera = new THREE.PerspectiveCamera(fov, w / h, CAMERA_NEAR, CAMERA_FAR)
        camera.position.set(0, 0, 3)
        cameraRef.current = camera

        // Load CDN Spark.js — imports "three" via browser importmap → same CDN URL →
        // same cached module instance → shared ShaderChunk (splatDefines etc.).
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        const SparkModule = await import(/* @vite-ignore */ SPARK_CDN) as unknown as typeof import('@sparkjsdev/spark')

        const { SparkRenderer, SplatMesh, modifiers } = SparkModule as unknown as {
          SparkRenderer: new (opts: { renderer: THREETypes.WebGLRenderer }) => THREETypes.Mesh & {
            update: (opts: { scene: THREETypes.Scene }) => void
            autoUpdate: boolean
          }
          SplatMesh: new (opts: { url?: string }) => THREETypes.Object3D & {
            initialized: Promise<unknown>
            worldModifier?: unknown
            enableWorldToView: boolean
            updateGenerator: () => void
          }
          modifiers: {
            setDepthColor: SparkDepthSource['setDepthColor']
          }
        }

        if (disposed) return

        const sparkRenderer = new SparkRenderer({ renderer })
        scene.add(sparkRenderer)
        sparkRef.current = sparkRenderer

        if (plyUrl) {
          const absoluteUrl = plyUrl.startsWith('http')
            ? plyUrl
            : `${window.location.origin}${plyUrl}`

          const splat = new SplatMesh({ url: absoluteUrl })
          // COLMAP/Nerfstudio convention: Y-down, Z-forward (OpenCV).
          // THREE.js convention:          Y-up,   Z-backward (OpenGL).
          // Rotating π around X simultaneously negates Y and Z, converting between them.
          ;(splat as THREETypes.Object3D & { rotation: { x: number } }).rotation.x = Math.PI
          scene.add(splat)
          splatRef.current = splat
          depthSourceRef.current = {
            splat,
            setDepthColor: modifiers.setDepthColor,
          }

          try {
            await splat.initialized
          } catch (e) {
            console.warn('Splat load warning:', e)
          }
        }

        if (!disposed) setReady(true)
      } catch (e) {
        console.error('Spark.js init error:', e)
        if (!disposed) {
          setError(String(e))
          setReady(true)
        }
      }
    })()

    return () => {
      disposed = true
      window.removeEventListener('resize', onResize)
      cancelAnimationFrame(rafRef.current)
      rendererRef.current?.dispose()
      rendererRef.current = null
      sceneRef.current    = null
      cameraRef.current   = null
      sparkRef.current    = null
      splatRef.current    = null
      depthSourceRef.current = null
    }
  }, [canvasRef, plyUrl, fov])

  return {
    renderer: rendererRef.current,
    scene:    sceneRef.current,
    camera:   cameraRef.current,
    depthSource: depthSourceRef.current,
    ready,
    error,
  }
}
