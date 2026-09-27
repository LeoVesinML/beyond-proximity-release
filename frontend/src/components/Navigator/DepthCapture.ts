/** Renderer-derived depth capture for Spark.js Gaussian splats. */
import * as THREE from 'three'

export const CAMERA_NEAR = 0.1
export const CAMERA_FAR = 100.0

export interface SparkDepthMesh {
  worldModifier?: unknown
  enableWorldToView: boolean
  updateGenerator: () => void
}

export interface SparkDepthSource {
  splat: SparkDepthMesh
  setDepthColor: (
    splat: SparkDepthMesh,
    minDepth: number,
    maxDepth: number,
    reverse?: boolean,
  ) => unknown
}

export interface DepthCaptureResult {
  depth: Float32Array
  width: number
  height: number
}

const MIN_ALPHA = 2 / 255

function decodeLogDepth(reverseGray: number, near: number, far: number): number {
  const normalized = 1 - Math.max(0, Math.min(1, reverseGray))
  const logNear = Math.log2(near + 1)
  const logFar = Math.log2(far + 1)
  return Math.pow(2, logNear + normalized * (logFar - logNear)) - 1
}

export class DepthCapture {
  capture(
    renderer: THREE.WebGLRenderer,
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
    source: SparkDepthSource,
  ): DepthCaptureResult {
    const width = renderer.domElement.width
    const height = renderer.domElement.height
    if (width <= 0 || height <= 0) {
      throw new Error(`Invalid render size ${width}x${height}`)
    }

    const pixels = new Uint8Array(width * height * 4)
    const savedTarget = renderer.getRenderTarget()
    const savedBackground = scene.background
    const savedClearColor = new THREE.Color()
    renderer.getClearColor(savedClearColor)
    const savedClearAlpha = renderer.getClearAlpha()
    const savedModifier = source.splat.worldModifier
    const savedWorldToView = source.splat.enableWorldToView

    try {
      // Spark's modifier encodes logarithmic view-space splat depth as reversed
      // grayscale. Transparent black remains an unobserved pixel.
      source.setDepthColor(source.splat, CAMERA_NEAR, CAMERA_FAR, true)
      scene.background = null
      renderer.setClearColor(0x000000, 0)
      renderer.setRenderTarget(null)
      renderer.clear(true, true, true)
      renderer.render(scene, camera)

      const gl = renderer.getContext()
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
    } finally {
      source.splat.worldModifier = savedModifier
      source.splat.enableWorldToView = savedWorldToView
      source.splat.updateGenerator()
      renderer.setRenderTarget(savedTarget)
      renderer.setClearColor(savedClearColor, savedClearAlpha)
      scene.background = savedBackground
      renderer.render(scene, camera)
    }

    const depth = new Float32Array(width * height)
    for (let y = 0; y < height; y += 1) {
      const sourceY = height - 1 - y // WebGL readback is bottom-up; PNG is top-down.
      for (let x = 0; x < width; x += 1) {
        const sourceIndex = (sourceY * width + x) * 4
        const alpha = pixels[sourceIndex + 3] / 255
        if (alpha <= MIN_ALPHA) continue

        const premultipliedGray = (
          pixels[sourceIndex] + pixels[sourceIndex + 1] + pixels[sourceIndex + 2]
        ) / (3 * 255)
        const reverseGray = Math.max(0, Math.min(1, premultipliedGray / alpha))
        depth[y * width + x] = decodeLogDepth(reverseGray, CAMERA_NEAR, CAMERA_FAR)
      }
    }
    return { depth, width, height }
  }

  encodeDepth(depthData: Float32Array): string {
    const bytes = new Uint8Array(depthData.buffer)
    let binary = ''
    const chunkSize = 0x8000
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize))
    }
    return btoa(binary)
  }

  dispose(): void {
    // No persistent GPU resources are allocated by this capture helper.
  }
}
