import { create } from 'zustand'
import type { QueryResult, TraversalEvent } from '../types'

interface QueryState {
  query: string
  isRunning: boolean
  events: TraversalEvent[]
  result: QueryResult | null
  visitedNodeIds: string[]

  setQuery: (q: string) => void
  setRunning: (v: boolean) => void
  pushEvent: (ev: TraversalEvent) => void
  setResult: (r: QueryResult | null) => void
  reset: () => void
}

export const useQueryStore = create<QueryState>((set) => ({
  query: '',
  isRunning: false,
  events: [],
  result: null,
  visitedNodeIds: [],

  setQuery: (q) => set({ query: q }),
  setRunning: (v) => set({ isRunning: v }),
  pushEvent: (ev) =>
    set((s) => ({
      events: [...s.events, ev],
      visitedNodeIds:
        ev.node_id && !s.visitedNodeIds.includes(ev.node_id)
          ? [...s.visitedNodeIds, ev.node_id]
          : s.visitedNodeIds,
    })),
  setResult: (r) => set({ result: r }),
  reset: () => set({ events: [], result: null, visitedNodeIds: [], isRunning: false }),
}))
