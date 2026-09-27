/**
 * HUD — heads-up display overlay for the 3D navigator.
 *
 * Shows: camera position/rotation, capture count, capture overlay, crosshair, hints.
 */
import React, { useEffect, useState } from 'react'
import * as THREE from 'three'
import { useSceneStore } from '../../store/sceneStore'

interface HUDProps {
  camera: THREE.PerspectiveCamera | null
}

export function HUD({ camera }: HUDProps) {
  const { capturedViews, isCapturing, sceneId } = useSceneStore()
  const [pos, setPos] = useState<[number, number, number]>([0, 0, 0])
  const [rot, setRot] = useState<[number, number, number]>([0, 0, 0])

  // Update position / rotation every frame
  useEffect(() => {
    let raf: number
    const tick = () => {
      if (camera) {
        const p = camera.position
        const e = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ')
        setPos([+p.x.toFixed(2), +p.y.toFixed(2), +p.z.toFixed(2)])
        setRot([
          +(e.x * (180 / Math.PI)).toFixed(1),
          +(e.y * (180 / Math.PI)).toFixed(1),
          +(e.z * (180 / Math.PI)).toFixed(1),
        ])
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [camera])

  return (
    <>
      {/* Capture flash — plain white overlay, visible only while the async capture
          is in flight. No animation loop; it disappears when isCapturing goes false. */}
      {isCapturing && (
        <div className="absolute inset-0 bg-white/40 pointer-events-none z-20" />
      )}

      {/* Crosshair */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
        <div className="relative">
          <div className="w-px h-4 bg-white opacity-70 mx-auto" />
          <div className="w-4 h-px bg-white opacity-70 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
        </div>
      </div>

      {/* Top-left info */}
      <div className="absolute top-3 left-3 text-xs font-mono text-green-400 bg-black/60 rounded p-2 space-y-0.5 pointer-events-none z-10">
        <div>Scene: <span className="text-white">{sceneId}</span></div>
        <div>Pos: <span className="text-white">{pos[0]}, {pos[1]}, {pos[2]}</span></div>
        <div>Rot: <span className="text-white">{rot[0]}°, {rot[1]}°</span></div>
        <div>Views: <span className="text-yellow-300">{capturedViews.length}</span></div>
      </div>

      {/* Bottom controls hint */}
      <div className="absolute bottom-3 left-3 text-xs font-mono text-gray-400 bg-black/50 rounded p-2 pointer-events-none z-10">
        <span className="text-white">W/A/S/D</span> move &nbsp;
        <span className="text-white">Q/E</span> up/down &nbsp;
        <span className="text-white">Shift</span> boost &nbsp;
        <span className="text-yellow-300">R</span> capture &nbsp;
        <span className="text-gray-500">(click to lock mouse)</span>
      </div>

      {/* Capture badge */}
      {isCapturing && (
        <div className="absolute top-3 right-3 text-xs font-bold text-red-400 bg-black/70 rounded px-2 py-1 z-10 animate-pulse">
          ● CAPTURING
        </div>
      )}
    </>
  )
}
