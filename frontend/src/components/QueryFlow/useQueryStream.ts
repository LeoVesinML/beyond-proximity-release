/**
 * useQueryStream — subscribes to the backend WebSocket to receive traversal events.
 */
import { useEffect, useRef } from 'react'
import { createQueryStream } from '../../api/client'
import { useQueryStore } from '../../store/queryStore'
import { useTreeStore } from '../../store/treeStore'
import type { TraversalEvent } from '../../types'

export function useQueryStream(sceneId: string) {
  const { pushEvent, setResult, setRunning } = useQueryStore()
  const { setActiveNode, setHighlightedViews } = useTreeStore()
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    wsRef.current = createQueryStream(sceneId, (raw) => {
      const ev = raw as TraversalEvent
      if ((ev as { echo?: unknown }).echo) return
      pushEvent(ev)

      switch (ev.event) {
        case 'visit_node':
          if (ev.node_id) setActiveNode(ev.node_id)
          break
        case 'leaf_check':
          if (ev.view_id) setHighlightedViews([ev.view_id])
          break
        case 'found':
        case 'not_found':
          setRunning(false)
          break
      }
    })

    return () => {
      wsRef.current?.close()
    }
  }, [sceneId, pushEvent, setResult, setRunning, setActiveNode, setHighlightedViews])

  return wsRef
}
