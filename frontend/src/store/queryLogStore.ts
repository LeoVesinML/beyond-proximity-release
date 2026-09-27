import { create } from 'zustand'
import type { QuerySession, QuerySessionHeader } from '../types'

interface QueryLogState {
  sessions: QuerySessionHeader[]
  activeSession: QuerySession | null
  pollingSessionId: string | null
  pendingOpenSessionId: string | null

  setSessions: (s: QuerySessionHeader[]) => void
  setActiveSession: (s: QuerySession | null) => void
  setPollingSessionId: (id: string | null) => void
  setPendingOpenSessionId: (id: string | null) => void
  updateActiveSession: (s: QuerySession) => void
}

export const useQueryLogStore = create<QueryLogState>((set) => ({
  sessions: [],
  activeSession: null,
  pollingSessionId: null,
  pendingOpenSessionId: null,

  setSessions: (sessions) => set({ sessions }),
  setActiveSession: (activeSession) => set({ activeSession }),
  setPollingSessionId: (pollingSessionId) => set({ pollingSessionId }),
  setPendingOpenSessionId: (pendingOpenSessionId) => set({ pendingOpenSessionId }),
  updateActiveSession: (s) => set({ activeSession: s }),
}))
