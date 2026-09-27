import { create } from 'zustand'
import type { D3TreeNode } from '../types'

interface TreeState {
  treeData: D3TreeNode | null
  activeNodeId: string | null
  highlightedViewIds: string[]
  isLoading: boolean

  setTreeData: (data: D3TreeNode | null) => void
  setActiveNode: (id: string | null) => void
  setHighlightedViews: (ids: string[]) => void
  setLoading: (v: boolean) => void
}

export const useTreeStore = create<TreeState>((set) => ({
  treeData: null,
  activeNodeId: null,
  highlightedViewIds: [],
  isLoading: false,

  setTreeData: (data) => set({ treeData: data }),
  setActiveNode: (id) => set({ activeNodeId: id }),
  setHighlightedViews: (ids) => set({ highlightedViewIds: ids }),
  setLoading: (v) => set({ isLoading: v }),
}))
