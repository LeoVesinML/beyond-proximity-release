/**
 * CapturedViewList — sidebar showing thumbnails of all captured views.
 */
import React from 'react'
import { useSceneStore } from '../../store/sceneStore'

export function CapturedViewList() {
  const { capturedViews } = useSceneStore()

  if (capturedViews.length === 0) {
    return (
      <div className="text-gray-500 text-xs p-3 text-center">
        No views captured yet.<br />
        <span className="text-yellow-300">Press R</span> inside the navigator.
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2 p-2 overflow-y-auto max-h-[calc(100vh-200px)]">
      {capturedViews.map((view) => (
        <div
          key={view.view_id}
          className="rounded overflow-hidden border border-gray-700 hover:border-green-500 transition-colors cursor-pointer"
          title={`Position: ${view.position.map((v) => v.toFixed(2)).join(', ')}`}
        >
          <img
            src={view.thumbnail}
            alt={view.view_id}
            className="w-full h-20 object-cover"
          />
          <div className="bg-gray-900 px-1.5 py-0.5 text-xs text-gray-400 font-mono flex justify-between">
            <span className="text-green-400">{view.view_id}</span>
            <span>{new Date(view.timestamp).toLocaleTimeString()}</span>
          </div>
        </div>
      ))}
    </div>
  )
}
