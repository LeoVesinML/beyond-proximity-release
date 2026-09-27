/**
 * Navigator — orchestrates Spark.js scene, FPS controls, depth capture, and HUD.
 *
 * This is the main 3D viewport component.  Drop a .ply URL into the `plyUrl`
 * prop to load a Gaussian Splatting scene.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useSparkScene } from './useSparkScene'
import { useFPSControls } from './useFPSControls'
import { useViewCapture } from './useViewCapture'
import { HUD } from './HUD'
import { CapturedViewList } from './CapturedViewList'

interface NavigatorProps {
  plyUrl: string | null
  showSidebar?: boolean
}

export function Navigator({ plyUrl, showSidebar = true }: NavigatorProps) {
  const canvasRef     = useRef<HTMLCanvasElement>(null)
  const containerRef  = useRef<HTMLDivElement>(null)

  const { renderer, scene, camera, depthSource, ready, error } = useSparkScene({
    canvasRef,
    plyUrl,
  })

  const { update: updateFPS } = useFPSControls({
    camera,
    domElement: canvasRef.current,
    enabled: ready,
  })

  // Wire FPS update into the animation loop via a ref callback
  useEffect(() => {
    if (!renderer || !camera) return
    let raf: number
    const loop = () => {
      raf = requestAnimationFrame(loop)
      updateFPS(camera)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [renderer, camera, updateFPS])

  const { captureError } = useViewCapture({
    renderer,
    scene,
    camera,
    canvas: canvasRef.current,
    depthSource,
    plyUrl,
    enabled: ready,
  })

  return (
    <div className="flex h-full w-full bg-gray-950">
      {/* ── 3D Viewport ────────────────────────────────────────────── */}
      <div ref={containerRef} className="relative flex-1 overflow-hidden">
        <canvas
          ref={canvasRef}
          className="w-full h-full block"
          style={{ touchAction: 'none' }}
        />

        {/* Loading overlay */}
        {!ready && !error && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-950/80 text-white">
            <div className="text-center space-y-2">
              <div className="animate-spin w-10 h-10 border-4 border-green-500 border-t-transparent rounded-full mx-auto" />
              <p className="text-sm text-gray-300">
                {plyUrl ? 'Loading Gaussian Splat…' : 'No scene loaded'}
              </p>
            </div>
          </div>
        )}

        {/* Error overlay */}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-950/80 text-red-400 p-8 text-center text-sm">
            <div>
              <p className="font-bold mb-1">Scene load error</p>
              <p className="text-gray-400 break-all">{error}</p>
              <p className="mt-2 text-gray-500">Navigator still active — fly around and capture anyway.</p>
            </div>
          </div>
        )}

        {captureError && (
          <div className="absolute bottom-14 right-3 max-w-md rounded bg-red-950/90 px-3 py-2 text-xs text-red-200 z-20">
            Capture rejected: {captureError}
          </div>
        )}

        <HUD camera={camera} />
      </div>

      {/* ── Captured Views Sidebar ──────────────────────────────────── */}
      {showSidebar && (
        <div className="w-44 bg-gray-900 border-l border-gray-800 flex-shrink-0">
          <div className="px-2 py-1.5 text-xs font-semibold text-gray-400 uppercase tracking-wider border-b border-gray-800">
            Captured Views
          </div>
          <CapturedViewList />
        </div>
      )}
    </div>
  )
}
