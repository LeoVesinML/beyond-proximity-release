/**
 * ControlPanel — scene selector, PLY URL input, and tree refresh button.
 */
import React, { useEffect, useState } from 'react'
import { healthApi, scenesApi, treeApi, queryApi, queryLogApi } from '../../api/client'
import { useSceneStore } from '../../store/sceneStore'
import { useTreeStore } from '../../store/treeStore'
import { useQueryStore } from '../../store/queryStore'
import { useQueryLogStore } from '../../store/queryLogStore'
import type { D3TreeNode } from '../../types'

interface ControlPanelProps {
  onPlyUrlChange: (url: string) => void
  plyUrl: string
  onQuerySubmitted?: (sessionId: string) => void
}

export function ControlPanel({ onPlyUrlChange, plyUrl, onQuerySubmitted }: ControlPanelProps) {
  const { sceneId, setSceneId, setSceneInfo } = useSceneStore()
  const { setTreeData, setLoading } = useTreeStore()
  const { setQuery, setRunning, setResult, reset: resetQuery } = useQueryStore()
  const { setPollingSessionId, setSessions, setPendingOpenSessionId } = useQueryLogStore()

  const [inputScene, setInputScene]   = useState(sceneId)
  const [inputPly,   setInputPly]     = useState(plyUrl)
  const [plyOptions, setPlyOptions]   = useState<string[]>([])
  const [status,     setStatus]       = useState<string | null>(null)
  const [queryText,  setQueryText]    = useState('')

  useEffect(() => {
    setInputPly(plyUrl)
  }, [plyUrl])

  const refreshPlyList = async (autoPick = false) => {
    try {
      const res = await healthApi.get()
      const files = res.data.ply_files ?? []
      setPlyOptions(files)
      if (files.length === 0) {
        setStatus('Нет .ply в scenes/ — положи файлы в папку scenes/')
        return
      }
      if (autoPick && !plyUrl) {
        const preferred = files.find((f) => f === 'ConferenceHall.ply') ?? files[0]
        const url = `/scenes/${preferred}`
        setInputPly(url)
        onPlyUrlChange(url)
      }
      setStatus(`Доступно сцен: ${files.length}`)
    } catch {
      setStatus('Backend не отвечает — проверь ./start_all.sh')
    }
  }

  useEffect(() => {
    refreshPlyList(true)
    const onFocus = () => refreshPlyList()
    const timer = window.setInterval(() => refreshPlyList(), 10_000)
    window.addEventListener('focus', onFocus)
    return () => {
      window.removeEventListener('focus', onFocus)
      window.clearInterval(timer)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const applyScene = async () => {
    setSceneId(inputScene)
    setStatus('Initialising scene…')
    try {
      await scenesApi.init(inputScene)
      const info = await scenesApi.get(inputScene)
      setSceneInfo(info.data)
      setStatus(`Scene "${inputScene}" ready`)
    } catch {
      setStatus('Scene init failed — check backend')
    }
  }

  const loadPly = () => {
    onPlyUrlChange(inputPly)
    setStatus('Loading PLY…')
  }

  const refreshTree = async () => {
    setLoading(true)
    try {
      const res = await treeApi.getViz(sceneId)
      setTreeData(res.data as D3TreeNode)
      setStatus('Tree refreshed')
    } catch {
      setStatus('No tree found — build it via Cursor AI MCP tools')
    } finally {
      setLoading(false)
    }
  }

  const submitQuery = async () => {
    if (!queryText.trim()) return
    resetQuery()
    const q = queryText.trim()
    setQuery(q)
    setRunning(true)
    setStatus(`Running query: "${q}"…`)
    try {
      const sessRes = await queryLogApi.runSession(sceneId, q)
      const sessionId = sessRes.data.session_id
      setPollingSessionId(sessionId)
      setPendingOpenSessionId(sessionId)
      onQuerySubmitted?.(sessionId)

      try {
        const listRes = await queryLogApi.listSessions(sceneId)
        setSessions(listRes.data.sessions)
      } catch { /* not critical */ }

      setStatus(`Session ${sessionId} — pipeline running…`)

      const poll = async (): Promise<void> => {
        try {
          for (let i = 0; i < 120; i += 1) {
            await new Promise((r) => setTimeout(r, 800))
            const res = await queryLogApi.getSession(sceneId, sessionId)
            if (!res.data.finished_at) continue

            const r = res.data.result
            setResult({
              query: q,
              found: r?.found ?? false,
              view_id: r?.view_id ?? null,
              bbox_3d: r?.bbox_3d ?? null,
              camera_pose: null,
              confidence: r?.confidence ?? 0,
              pipeline_phase: r?.found ? 'found' : 'not_found',
              explanation: r?.explanation ?? 'No result.',
            })
            setStatus(
              r?.found
                ? `Found in ${r.view_id} (session ${sessionId})`
                : `Not found (session ${sessionId})`,
            )
            try {
              const listRes = await queryLogApi.listSessions(sceneId)
              setSessions(listRes.data.sessions)
            } catch { /* not critical */ }
            return
          }
          setStatus(`Session ${sessionId} timed out — open Query Flow tab`)
        } catch {
          setStatus('Lost connection to session — check backend')
        } finally {
          setRunning(false)
        }
      }

      void poll()
    } catch {
      setResult(null)
      setStatus('Query failed — check backend is running')
      setRunning(false)
    }
  }

  return (
    <div className="bg-gray-900 border-b border-gray-800 px-4 py-2 flex flex-wrap items-center gap-3">
      {/* Scene */}
      <div className="flex items-center gap-1.5">
        <label className="text-xs text-gray-400 whitespace-nowrap">Scene ID</label>
        <input
          className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white w-28 focus:outline-none focus:border-green-500"
          value={inputScene}
          onChange={(e) => setInputScene(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && applyScene()}
          placeholder="default"
        />
        <button
          onClick={applyScene}
          className="text-xs bg-green-700 hover:bg-green-600 text-white px-2 py-1 rounded transition-colors"
        >
          Set
        </button>
      </div>

      {/* PLY scene */}
      <div className="flex items-center gap-1.5 flex-1 min-w-[260px]">
        <label className="text-xs text-gray-400 whitespace-nowrap">3D Scene</label>
        {plyOptions.length > 0 ? (
          <select
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white flex-1 focus:outline-none focus:border-green-500"
            value={inputPly}
            onChange={(e) => {
              setInputPly(e.target.value)
              onPlyUrlChange(e.target.value)
              setStatus('Loading PLY…')
            }}
          >
            {plyOptions.map((f) => (
              <option key={f} value={`/scenes/${f}`}>
                {f}
              </option>
            ))}
          </select>
        ) : (
          <input
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white flex-1 focus:outline-none focus:border-green-500"
            value={inputPly}
            onChange={(e) => setInputPly(e.target.value)}
            placeholder="/scenes/ConferenceHall.ply"
          />
        )}
        <button
          onClick={loadPly}
          className="text-xs bg-indigo-700 hover:bg-indigo-600 text-white px-2 py-1 rounded transition-colors"
        >
          Load
        </button>
      </div>

      {/* Tree */}
      <button
        onClick={refreshTree}
        className="text-xs bg-yellow-700 hover:bg-yellow-600 text-white px-2 py-1 rounded transition-colors whitespace-nowrap"
      >
        ↻ Refresh Tree
      </button>

      {/* Query */}
      <div className="flex items-center gap-1.5 flex-1 min-w-[220px]">
        <label className="text-xs text-gray-400 whitespace-nowrap">Query</label>
        <input
          className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white flex-1 focus:outline-none focus:border-blue-500"
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submitQuery()}
          placeholder="Find the red sofa…"
        />
        <button
          onClick={submitQuery}
          className="text-xs bg-blue-700 hover:bg-blue-600 text-white px-2 py-1 rounded transition-colors"
        >
          Ask
        </button>
      </div>

      {/* Status */}
      {status && (
        <span className="text-xs text-gray-400 italic truncate max-w-xs">{status}</span>
      )}
    </div>
  )
}
