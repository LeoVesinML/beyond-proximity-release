import { create } from 'zustand'
import type { CapturedView, SceneInfo } from '../types'

interface SceneState {
  sceneId: string
  sceneInfo: SceneInfo | null
  capturedViews: CapturedView[]
  nextViewId: number
  isCapturing: boolean

  setSceneId: (id: string) => void
  setSceneInfo: (info: SceneInfo | null) => void
  addCapturedView: (view: CapturedView) => void
  setIsCapturing: (v: boolean) => void
  reset: () => void
}

export const useSceneStore = create<SceneState>((set) => ({
  sceneId: 'default',
  sceneInfo: null,
  capturedViews: [],
  nextViewId: 1,
  isCapturing: false,

  setSceneId: (id) => set({
    sceneId: id,
    sceneInfo: null,
    capturedViews: [],
    nextViewId: 1,
  }),
  setSceneInfo: (info) => set(() => {
    const existingIds = info?.frames
      .map((frame) => /^v(\d+)$/.exec(frame.view_id)?.[1])
      .filter((value): value is string => value !== undefined)
      .map(Number) ?? []
    return {
      sceneInfo: info,
      nextViewId: existingIds.length > 0 ? Math.max(...existingIds) + 1 : 1,
    }
  }),
  addCapturedView: (view) =>
    set((s) => ({
      capturedViews: [...s.capturedViews, view],
      nextViewId: s.nextViewId + 1,
    })),
  setIsCapturing: (v) => set({ isCapturing: v }),
  reset: () => set({ capturedViews: [], nextViewId: 1, sceneInfo: null }),
}))
