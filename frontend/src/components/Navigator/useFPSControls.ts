/**
 * useFPSControls — first-person shooter camera controls.
 *
 * Controls:
 *   WASD      → forward / strafe
 *   Q / E     → down / up
 *   Shift     → 3× speed boost
 *   Mouse     → look (pointer lock)
 */
import { useEffect, useRef } from 'react'
import * as THREE from 'three'

const BASE_SPEED   = 0.05   // units per frame
const BOOST_MULT   = 3.0
const LOOK_SENS    = 0.002  // radians per pixel

interface UseFPSControlsOptions {
  camera: THREE.PerspectiveCamera | null
  domElement: HTMLElement | null
  enabled?: boolean
}

export function useFPSControls({ camera, domElement, enabled = true }: UseFPSControlsOptions) {
  const keys   = useRef<Set<string>>(new Set())
  const pitch  = useRef(0)   // radians, clamped ±89°
  const yaw    = useRef(0)   // radians

  // ── Keyboard ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!enabled) return
    const onDown = (e: KeyboardEvent) => keys.current.add(e.code)
    const onUp   = (e: KeyboardEvent) => keys.current.delete(e.code)
    window.addEventListener('keydown', onDown)
    window.addEventListener('keyup',   onUp)
    return () => {
      window.removeEventListener('keydown', onDown)
      window.removeEventListener('keyup',   onUp)
    }
  }, [enabled])

  // ── Pointer lock + mouse look ─────────────────────────────────────────────
  useEffect(() => {
    if (!domElement || !enabled) return

    const onClick = () => domElement.requestPointerLock()
    const onMouseMove = (e: MouseEvent) => {
      if (document.pointerLockElement !== domElement) return
      yaw.current   -= e.movementX * LOOK_SENS
      pitch.current -= e.movementY * LOOK_SENS
      const limit = (89 * Math.PI) / 180
      pitch.current = Math.max(-limit, Math.min(limit, pitch.current))
    }

    domElement.addEventListener('click',     onClick)
    document.addEventListener('mousemove', onMouseMove)
    return () => {
      domElement.removeEventListener('click',     onClick)
      document.removeEventListener('mousemove', onMouseMove)
    }
  }, [domElement, enabled])

  // ── Per-frame update (called from Navigator animation loop) ───────────────
  const update = (camera: THREE.PerspectiveCamera) => {
    if (!enabled) return

    // Apply orientation from yaw + pitch (Euler: YXZ order)
    const euler = new THREE.Euler(pitch.current, yaw.current, 0, 'YXZ')
    camera.quaternion.setFromEuler(euler)

    // Movement in local camera space
    const k = keys.current
    const boost  = k.has('ShiftLeft') || k.has('ShiftRight') ? BOOST_MULT : 1
    const speed  = BASE_SPEED * boost

    const forward = new THREE.Vector3()
    const right   = new THREE.Vector3()
    const up      = new THREE.Vector3(0, 1, 0)

    camera.getWorldDirection(forward)
    forward.y = 0
    forward.normalize()
    right.crossVectors(forward, up).normalize()

    if (k.has('KeyW') || k.has('ArrowUp'))    camera.position.addScaledVector(forward,  speed)
    if (k.has('KeyS') || k.has('ArrowDown'))  camera.position.addScaledVector(forward, -speed)
    if (k.has('KeyA') || k.has('ArrowLeft'))  camera.position.addScaledVector(right,   -speed)
    if (k.has('KeyD') || k.has('ArrowRight')) camera.position.addScaledVector(right,    speed)
    if (k.has('KeyQ'))                        camera.position.y -= speed
    if (k.has('KeyE'))                        camera.position.y += speed
  }

  return { update }
}
