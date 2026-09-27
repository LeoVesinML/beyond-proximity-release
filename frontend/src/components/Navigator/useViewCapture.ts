/**
 * useViewCapture — handles "R" keypress to capture current view.
 *
 * On "R":
 *   1. Take RGB screenshot via canvas.toDataURL
 *   2. Capture linearised depth via DepthCapture
 *   3. Extract 4×4 camera-to-world from camera.matrixWorld
 *   4. POST CapturePayload to /api/captures/save
 *   5. Add thumbnail to scene store
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { DepthCapture, CAMERA_NEAR, CAMERA_FAR, type SparkDepthSource } from './DepthCapture'
import { captureApi } from '../../api/client'
import { useSceneStore } from '../../store/sceneStore'

interface UseViewCaptureOptions {
  renderer: THREE.WebGLRenderer | null
  scene: THREE.Scene | null
  camera: THREE.PerspectiveCamera | null
  canvas: HTMLCanvasElement | null
  depthSource: SparkDepthSource | null
  plyUrl: string | null
  enabled?: boolean
}

function getTransformMatrix(camera: THREE.PerspectiveCamera): number[][] {
  camera.updateMatrixWorld()
  const m = camera.matrixWorld.elements  // column-major
  return [
    [m[0], m[4], m[8],  m[12]],
    [m[1], m[5], m[9],  m[13]],
    [m[2], m[6], m[10], m[14]],
    [m[3], m[7], m[11], m[15]],
  ]
}

function canvasToBase64Png(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL('image/png')
}

function getIntrinsics(camera: THREE.PerspectiveCamera, width: number, height: number) {
  camera.updateProjectionMatrix()
  const projection = camera.projectionMatrix.elements
  return {
    fl_x: Math.abs(projection[0]) * width / 2,
    fl_y: Math.abs(projection[5]) * height / 2,
    cx: (1 - projection[8]) * width / 2,
    cy: (1 + projection[9]) * height / 2,
  }
}

function depthStats(depth: Float32Array) {
  let validCount = 0
  let minimum = Number.POSITIVE_INFINITY
  let maximum = Number.NEGATIVE_INFINITY
  for (const value of depth) {
    if (!Number.isFinite(value) || value <= 0) continue
    validCount += 1
    minimum = Math.min(minimum, value)
    maximum = Math.max(maximum, value)
  }
  if (validCount === 0) throw new Error('Depth capture contains no positive finite renderer depths.')
  if (Math.abs(maximum - minimum) <= 1e-6) {
    throw new Error(`Depth capture is constant (${minimum}); capture was not saved.`)
  }
  return { validCount, minimum, maximum }
}

export function useViewCapture({
  renderer,
  scene,
  camera,
  canvas,
  depthSource,
  plyUrl,
  enabled = true,
}: UseViewCaptureOptions) {
  const depthCaptureRef  = useRef<DepthCapture | null>(null)
  const capturingRef     = useRef(false)           // ref guard — immune to stale closures
  const { sceneId, nextViewId, addCapturedView, setIsCapturing } = useSceneStore()
  const [captureError, setCaptureError] = useState<string | null>(null)

  // Lazily create or resize DepthCapture when canvas changes
  useEffect(() => {
    if (!canvas) return
    depthCaptureRef.current?.dispose()
    depthCaptureRef.current = new DepthCapture()
    return () => {
      depthCaptureRef.current?.dispose()
    }
  }, [canvas])

  const capture = useCallback(async () => {
    if (!renderer || !scene || !camera || !canvas || !depthSource) return
    if (capturingRef.current) return   // prevent re-entry if R is pressed mid-capture
    capturingRef.current = true
    setIsCapturing(true)
    setCaptureError(null)

    try {
      const viewId = `v${String(nextViewId).padStart(3, '0')}`
      const w = renderer.domElement.width
      const h = renderer.domElement.height

      // RGB
      const rgb_b64 = canvasToBase64Png(canvas)

      // Depth
      if (!depthCaptureRef.current) {
        depthCaptureRef.current = new DepthCapture()
      }
      const depthResult = depthCaptureRef.current.capture(renderer, scene, camera, depthSource)
      if (depthResult.width !== w || depthResult.height !== h) {
        throw new Error('RGB/depth render dimensions do not match.')
      }
      const stats = depthStats(depthResult.depth)
      const depth_b64 = depthCaptureRef.current.encodeDepth(depthResult.depth)

      // Camera pose
      const transform_matrix = getTransformMatrix(camera)
      const pos = camera.position

      const { fl_x, fl_y, cx, cy } = getIntrinsics(camera, w, h)
      const captured_at_utc = new Date().toISOString()

      await captureApi.save({
        scene_id: sceneId,
        view_id: viewId,
        rgb_b64,
        depth_b64,
        transform_matrix,
        fl_x,
        fl_y,
        cx,
        cy,
        width: w,
        height: h,
        source_ply_url: plyUrl,
        captured_at_utc,
        pose_coordinate_convention: 'threejs_world_camera_to_world_rh_y_up_camera_forward_minus_z',
        depth_source: 'spark_setDepthColor_log_view_space_splat_center',
        depth_near: CAMERA_NEAR,
        depth_far: CAMERA_FAR,
        depth_valid_pixel_count: stats.validCount,
        depth_min: stats.minimum,
        depth_max: stats.maximum,
      })

      addCapturedView({
        view_id: viewId,
        thumbnail: rgb_b64,
        position: [pos.x, pos.y, pos.z],
        timestamp: Date.now(),
      })
    } catch (err) {
      console.error('Capture failed', err)
      setCaptureError(err instanceof Error ? err.message : String(err))
    } finally {
      capturingRef.current = false
      setIsCapturing(false)
    }
  }, [renderer, scene, camera, canvas, depthSource, plyUrl, sceneId, nextViewId, addCapturedView, setIsCapturing])

  // "R" key handler
  useEffect(() => {
    if (!enabled) return
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'KeyR' && !e.repeat) capture()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [enabled, capture])

  return { capture, captureError }
}
