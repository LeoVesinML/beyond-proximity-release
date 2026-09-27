/**
 * QueryFlow — full §7 pipeline visualisation.
 *
 * Left column: session history list.
 * Right column: active session trace — one card per pipeline step.
 *   • Decomposition: shows structured_plan fields
 *   • Traversal: shows path nodes + descend_into + reasoning
 *   • Leaf check: found / not-found badge, matched object, bbox
 *   • Result: final verdict banner + bbox_3d if available
 */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { queryLogApi, annotateApi } from '../../api/client'
import { useSceneStore } from '../../store/sceneStore'
import { useQueryLogStore } from '../../store/queryLogStore'
import type {
  QueryStep, QueryStepDecomposition, QueryStepTraversal,
  QueryStepLeafCheck, QueryStepResult, QueryStepError, QuerySessionHeader,
} from '../../types'

// ── Helpers ──────────────────────────────────────────────────────────────────

function relTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  if (diff < 60_000) return `${Math.round(diff / 1000)}s ago`
  if (diff < 3_600_000) return `${Math.round(diff / 60_000)}m ago`
  return `${Math.round(diff / 3_600_000)}h ago`
}

function Badge({ label, color }: { label: string; color: string }) {
  return (
    <span className={`inline-block text-[10px] font-bold px-1.5 py-0.5 rounded ${color}`}>
      {label}
    </span>
  )
}

// ── Step cards ────────────────────────────────────────────────────────────────

function DecompCard({ step }: { step: QueryStepDecomposition }) {
  const plan = step.structured_plan
  return (
    <div className="border border-indigo-700 rounded-lg p-3 bg-indigo-950/40 space-y-1.5">
      <div className="flex items-center gap-2">
        <Badge label="DECOMPOSE" color="bg-indigo-600 text-white" />
        <span className="text-[10px] text-gray-500">{relTime(step.ts)}</span>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[11px]">
        {Object.entries(plan).filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => (
          <React.Fragment key={k}>
            <span className="text-gray-500">{k}</span>
            <span className="text-gray-200 truncate" title={String(v)}>
              {typeof v === 'object' ? JSON.stringify(v) : String(v)}
            </span>
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}

function TraversalCard({ step }: { step: QueryStepTraversal }) {
  return (
    <div className="border border-sky-800 rounded-lg p-3 bg-sky-950/30 space-y-1.5">
      <div className="flex items-center gap-2">
        <Badge label="TRAVERSE" color="bg-sky-600 text-white" />
        <span className="text-white text-xs font-medium">{step.node_name}</span>
        <span className="text-[10px] text-gray-500 ml-auto">{relTime(step.ts)}</span>
      </div>
      <div className="text-[11px] text-gray-400">
        {step.children_count} children considered: {step.children_names.join(', ')}
      </div>
      <div className="flex flex-wrap gap-1">
        {step.children_names.map((name, i) => {
          const nid = step.descend_into.find(d =>
            name.toLowerCase().includes(d.toLowerCase()) ||
            d.toLowerCase().includes(name.toLowerCase().replace(/\s+/g, '_'))
          )
          const chosen = step.descend_into.some(d =>
            d.replace(/_/g, ' ').toLowerCase() === name.toLowerCase() || d === nid
          )
          return (
            <span
              key={i}
              className={`text-[10px] px-2 py-0.5 rounded-full border ${
                chosen
                  ? 'bg-sky-600/30 border-sky-500 text-sky-200'
                  : 'bg-gray-800 border-gray-700 text-gray-500 line-through'
              }`}
            >
              {name}
            </span>
          )
        })}
      </div>
      {step.reasoning && (
        <p className="text-[11px] text-gray-400 italic">"{step.reasoning}"</p>
      )}
    </div>
  )
}

function LeafCard({ step, sceneId }: { step: QueryStepLeafCheck; sceneId: string }) {
  const confColor = step.confidence === 'high'
    ? 'text-emerald-400' : step.confidence === 'medium'
      ? 'text-yellow-400' : 'text-gray-400'

  const [annotatedImg, setAnnotatedImg] = useState<string | null>(step.annotated_image ?? null)
  const [imgLoading, setImgLoading] = useState(false)
  const [rawImgFailed, setRawImgFailed] = useState(false)
  const rawImageUrl = `/scenes/${encodeURIComponent(sceneId)}/images/${encodeURIComponent(step.view_id)}.png`

  // Auto-fetch annotated image when the card first renders with a bbox and found=true
  useEffect(() => {
    if (!step.found || !step.bbox_2d || annotatedImg) return
    let cancelled = false
    setImgLoading(true)
    annotateApi.annotateView(sceneId, step.view_id, [{
      bbox_2d: step.bbox_2d,
      label: step.matched_object ?? 'match',
      confidence: step.confidence === 'high' ? 0.9 : step.confidence === 'medium' ? 0.65 : 0.4,
      color: '#FF4444',
    }]).then(res => {
      if (!cancelled) setAnnotatedImg(res.data.annotated_image)
    }).catch(() => { /* silently skip */ }).finally(() => {
      if (!cancelled) setImgLoading(false)
    })
    return () => { cancelled = true }
  }, [step.found, step.bbox_2d, step.view_id, annotatedImg, sceneId, step.matched_object, step.confidence])

  return (
    <div className={`border rounded-lg p-3 space-y-1.5 ${
      step.found ? 'border-emerald-700 bg-emerald-950/30' : 'border-gray-700 bg-gray-900/40'
    }`}>
      <div className="flex items-center gap-2">
        <Badge
          label={step.found ? '✓ FOUND' : '✗ MISS'}
          color={step.found ? 'bg-emerald-600 text-white' : 'bg-gray-700 text-gray-300'}
        />
        <span className="text-white text-xs font-mono">{step.view_id}</span>
        <span className={`text-[10px] ${confColor} ml-auto`}>{step.confidence}</span>
        <span className="text-[10px] text-gray-500">{relTime(step.ts)}</span>
      </div>
      <p className="text-[11px] text-gray-400">Leaf: <span className="text-gray-300">{step.node_name}</span></p>
      {step.matched_object && (
        <p className="text-[11px] text-emerald-300">Match: {step.matched_object}</p>
      )}
      {step.bbox_2d && (
        <p className="text-[10px] font-mono text-yellow-300">
          bbox₂D [{step.bbox_2d.map(v => v.toFixed(3)).join(', ')}]
        </p>
      )}
      {step.explanation && (
        <p className="text-[11px] text-gray-400 italic">"{step.explanation}"</p>
      )}
      {/* Annotated image */}
      {imgLoading && (
        <div className="text-[10px] text-gray-500 animate-pulse">Loading annotated image…</div>
      )}
      {annotatedImg && (
        <div className="mt-2 rounded overflow-hidden border border-emerald-800">
          <img
            src={annotatedImg}
            alt={`${step.view_id} annotated`}
            className="w-full object-contain max-h-72"
          />
        </div>
      )}
      {step.found && !step.bbox_2d && !annotatedImg && !rawImgFailed && (
        <div className="mt-2 rounded overflow-hidden border border-gray-700">
          <img
            src={rawImageUrl}
            alt={`${step.view_id} raw`}
            className="w-full object-contain max-h-72"
            onError={() => setRawImgFailed(true)}
          />
          <div className="bg-gray-900 px-2 py-1 text-[10px] text-gray-400">
            Raw view shown because this match has no bbox_2d annotation.
          </div>
        </div>
      )}
    </div>
  )
}

function ResultCard({ step }: { step: QueryStepResult }) {
  const found = step.step_type === 'found'
  return (
    <div className={`border-2 rounded-lg p-3 space-y-1.5 ${
      found ? 'border-emerald-500 bg-emerald-900/30' : 'border-red-700 bg-red-900/20'
    }`}>
      <div className="flex items-center gap-2">
        <Badge
          label={found ? '✓ RESULT: FOUND' : '✗ RESULT: NOT FOUND'}
          color={found ? 'bg-emerald-500 text-white' : 'bg-red-700 text-white'}
        />
        {step.view_id && <span className="text-white text-xs font-mono">{step.view_id}</span>}
        <span className="text-[10px] text-gray-500 ml-auto">{relTime(step.ts)}</span>
      </div>
      {step.explanation && (
        <p className="text-xs text-gray-300">{step.explanation}</p>
      )}
      {step.bbox_3d && (
        <p className="text-[11px] font-mono text-yellow-300">
          3D centre: [{(step.bbox_3d as any).center?.map((v: number) => v.toFixed(2)).join(', ')}]
        </p>
      )}
      {typeof step.confidence === 'number' && step.confidence > 0 && (
        <p className="text-[10px] text-gray-400">Confidence: {(step.confidence * 100).toFixed(0)}%</p>
      )}
    </div>
  )
}

function ErrorCard({ step }: { step: QueryStepError }) {
  return (
    <div className="border border-red-800 rounded-lg p-3 bg-red-950/30">
      <Badge label="ERROR" color="bg-red-700 text-white" />
      <p className="mt-1 text-xs text-red-300">{step.message}</p>
    </div>
  )
}

function StepCard({ step, sceneId }: { step: QueryStep; sceneId: string }) {
  switch (step.step_type) {
    case 'decomposition': return <DecompCard step={step as QueryStepDecomposition} />
    case 'traversal':     return <TraversalCard step={step as QueryStepTraversal} />
    case 'leaf_check':    return <LeafCard step={step as QueryStepLeafCheck} sceneId={sceneId} />
    case 'found':
    case 'not_found':     return <ResultCard step={step as QueryStepResult} />
    case 'error':         return <ErrorCard step={step as QueryStepError} />
    default:              return null
  }
}

// ── Tree-path timeline ────────────────────────────────────────────────────────

function TreePathTimeline({ steps }: { steps: QueryStep[] }) {
  const traversals = steps.filter(s => s.step_type === 'traversal') as QueryStepTraversal[]
  const leafChecks = steps.filter(s => s.step_type === 'leaf_check') as QueryStepLeafCheck[]
  const result = steps.find(s => s.step_type === 'found' || s.step_type === 'not_found') as QueryStepResult | undefined

  if (traversals.length === 0 && leafChecks.length === 0) return null

  const pathNodes = [
    ...traversals.map(t => ({ id: t.node_id, label: t.node_name, type: 'traversal' as const, descend: t.descend_into })),
    ...leafChecks.map(l => ({ id: l.view_id, label: `${l.node_name} → ${l.view_id}`, type: 'leaf' as const, found: l.found })),
  ]

  return (
    <div className="px-3 py-2 border-b border-gray-800 overflow-x-auto">
      <p className="text-[10px] text-gray-500 mb-1.5 uppercase tracking-wider">Traversal path</p>
      <div className="flex items-center gap-0 min-w-max">
        {pathNodes.map((n, i) => (
          <React.Fragment key={i}>
            <div className={`flex flex-col items-center gap-0.5 px-2 py-1 rounded text-center min-w-[72px] ${
              n.type === 'leaf'
                ? (n as any).found ? 'bg-emerald-900/40 border border-emerald-700' : 'bg-gray-800 border border-gray-700'
                : 'bg-sky-900/30 border border-sky-800'
            }`}>
              <span className={`text-[9px] font-bold uppercase ${
                n.type === 'leaf' ? ((n as any).found ? 'text-emerald-400' : 'text-gray-500') : 'text-sky-400'
              }`}>{n.type}</span>
              <span className="text-[10px] text-white leading-tight text-center max-w-[68px] truncate" title={n.label}>
                {n.label.length > 14 ? n.label.slice(0, 13) + '…' : n.label}
              </span>
            </div>
            {i < pathNodes.length - 1 && (
              <div className="w-4 h-px bg-gray-600 flex-shrink-0" />
            )}
          </React.Fragment>
        ))}
        {result && (
          <>
            <div className="w-4 h-px bg-gray-600 flex-shrink-0" />
            <div className={`px-2 py-1 rounded border text-center ${
              result.step_type === 'found'
                ? 'bg-emerald-900/50 border-emerald-500 text-emerald-300'
                : 'bg-red-900/30 border-red-700 text-red-400'
            }`}>
              <span className="text-[10px] font-bold">
                {result.step_type === 'found' ? '✓ Found' : '✗ Miss'}
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ── Session history sidebar ───────────────────────────────────────────────────

function SessionListItem({
  header, active, onClick,
}: { header: QuerySessionHeader; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2 rounded text-xs transition-colors space-y-0.5 ${
        active ? 'bg-indigo-900/50 border border-indigo-700' : 'hover:bg-gray-800 border border-transparent'
      }`}
    >
      <div className="flex items-center gap-1.5">
        {header.found === true && <span className="text-emerald-400 text-[9px]">●</span>}
        {header.found === false && header.finished_at && <span className="text-red-400 text-[9px]">●</span>}
        {header.found === null && <span className="text-yellow-400 text-[9px] animate-pulse">●</span>}
        <span className="text-white font-medium truncate" title={header.original_query}>
          {header.original_query.length > 26 ? header.original_query.slice(0, 25) + '…' : header.original_query}
        </span>
      </div>
      <div className="flex items-center justify-between text-[10px] text-gray-500">
        <span>{header.step_count} steps</span>
        <span>{relTime(header.started_at)}</span>
      </div>
    </button>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

interface QueryFlowProps {
  sceneId: string
}

export function QueryFlow({ sceneId }: QueryFlowProps) {
  const { sceneId: storeSceneId } = useSceneStore()
  const sid = sceneId || storeSceneId

  const {
    sessions,
    activeSession,
    pollingSessionId,
    pendingOpenSessionId,
    setSessions,
    setActiveSession,
    setPollingSessionId,
    setPendingOpenSessionId,
    updateActiveSession,
  } = useQueryLogStore()

  const logRef = useRef<HTMLDivElement>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Load session list
  const loadSessions = useCallback(async () => {
    try {
      const res = await queryLogApi.listSessions(sid)
      setSessions(res.data.sessions)
    } catch { /* no sessions yet */ }
  }, [sid, setSessions])

  useEffect(() => {
    setActiveSession(null)
    setPollingSessionId(null)
    void loadSessions()
  }, [sid, loadSessions, setActiveSession, setPollingSessionId])

  // Poll active session if it's still running
  useEffect(() => {
    if (!pollingSessionId) {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
      return
    }
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const res = await queryLogApi.getSession(sid, pollingSessionId)
        updateActiveSession(res.data)
        if (res.data.finished_at) {
          setPollingSessionId(null)
          void loadSessions()
        }
      } catch { setPollingSessionId(null) }
    }, 1500)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [pollingSessionId, sid, updateActiveSession, setPollingSessionId, loadSessions])

  // Auto-scroll step list
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [activeSession?.steps.length])

  const openSession = useCallback(async (sessionId: string) => {
    try {
      const res = await queryLogApi.getSession(sid, sessionId)
      setActiveSession(res.data)
      if (!res.data.finished_at) {
        setPollingSessionId(sessionId)
      } else {
        setPollingSessionId(null)
      }
    } catch { /* ignore */ }
  }, [sid, setActiveSession, setPollingSessionId])

  useEffect(() => {
    if (!pendingOpenSessionId) return
    void openSession(pendingOpenSessionId)
    setPendingOpenSessionId(null)
  }, [pendingOpenSessionId, openSession, setPendingOpenSessionId])

  return (
    <div className="flex h-full bg-gray-950 rounded-lg border border-gray-800 overflow-hidden">
      {/* ── Session sidebar ─────────────────────────────────────────── */}
      <div className="w-52 flex-shrink-0 flex flex-col border-r border-gray-800 bg-gray-900/50">
        <div className="px-3 py-2 border-b border-gray-800 flex items-center justify-between">
          <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">Sessions</span>
          <button onClick={loadSessions} className="text-[10px] text-gray-600 hover:text-gray-300" title="Refresh">↻</button>
        </div>
        <div className="flex-1 overflow-y-auto p-1.5 space-y-1">
          {sessions.length === 0 ? (
            <p className="text-[10px] text-gray-600 text-center py-4">
              No sessions yet.
              <br />Ask a question from
              <br />Cursor IDE.
            </p>
          ) : (
            sessions.map(h => (
              <SessionListItem
                key={h.session_id}
                header={h}
                active={activeSession?.session_id === h.session_id}
                onClick={() => openSession(h.session_id)}
              />
            ))
          )}
        </div>
      </div>

      {/* ── Active session panel ─────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        {activeSession ? (
          <>
            {/* Header */}
            <div className="flex items-center gap-2 px-3 py-2 bg-gray-900 border-b border-gray-800 flex-shrink-0">
              <span className="text-xs text-indigo-300 font-medium italic truncate">
                "{activeSession.original_query}"
              </span>
              {!activeSession.finished_at && (
                <span className="text-[10px] text-yellow-400 animate-pulse ml-auto">● Running</span>
              )}
              {activeSession.finished_at && (
                <span className="text-[10px] text-gray-500 ml-auto">
                  {activeSession.steps.length} steps · {relTime(activeSession.started_at)}
                </span>
              )}
            </div>

            {/* Tree-path timeline */}
            <TreePathTimeline steps={activeSession.steps} />

            {/* Step cards */}
            <div ref={logRef} className="flex-1 overflow-y-auto p-3 space-y-2 min-h-0">
              {activeSession.steps.map((step, i) => (
                <StepCard key={i} step={step} sceneId={sid} />
              ))}
              {!activeSession.finished_at && (
                <div className="flex items-center gap-2 text-[11px] text-gray-500 animate-pulse py-2">
                  <span className="w-2 h-2 rounded-full bg-yellow-500 animate-ping inline-block" />
                  Waiting for next step from Cursor…
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-center text-gray-600 text-xs p-8 space-y-2">
            <p className="text-2xl">🔍</p>
            <p className="text-sm text-gray-500">No session selected</p>
            <p>
              Ask a question in Cursor IDE.<br />
              The agent will run the §7 pipeline and<br />
              log every step here automatically.
            </p>
            {sessions.length > 0 && (
              <p className="text-gray-500">← Pick a session from the list</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
