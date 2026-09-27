/**
 * useTreeLayout — convert D3TreeNode hierarchy into D3 tree layout data.
 */
import { useMemo } from 'react'
import * as d3 from 'd3'
import type { D3TreeNode } from '../../types'

export interface LayoutNode {
  id: string
  name: string
  type: string
  summary: string
  x: number
  y: number
  parent: LayoutNode | null
  children: LayoutNode[]
  depth: number
  view_ids: string[]
}

export interface LayoutLink {
  source: LayoutNode
  target: LayoutNode
}

interface TreeLayoutResult {
  nodes: LayoutNode[]
  links: LayoutLink[]
  width: number
  height: number
}

export function useTreeLayout(
  treeData: D3TreeNode | null,
  containerWidth = 800,
  containerHeight = 500,
): TreeLayoutResult {
  return useMemo(() => {
    if (!treeData) return { nodes: [], links: [], width: containerWidth, height: containerHeight }

    const root = d3.hierarchy(treeData, (d) => d.children)
    const treeLayout = d3.tree<D3TreeNode>()
      .size([containerWidth - 80, containerHeight - 80])

    treeLayout(root as d3.HierarchyNode<D3TreeNode>)

    const nodes: LayoutNode[] = (root as d3.HierarchyNode<D3TreeNode>).descendants().map((d) => ({
      id: d.data.id,
      name: d.data.name,
      type: d.data.type,
      summary: d.data.summary,
      x: (d as d3.HierarchyPointNode<D3TreeNode>).x + 40,
      y: (d as d3.HierarchyPointNode<D3TreeNode>).y + 40,
      parent: null,
      children: [],
      depth: d.depth,
      view_ids: d.data.view_ids ?? [],
    }))

    // Wire parent/children refs
    const nodeMap = new Map(nodes.map((n) => [n.id, n]))
    ;(root as d3.HierarchyNode<D3TreeNode>).descendants().forEach((d) => {
      if (d.parent) {
        const node   = nodeMap.get(d.data.id)!
        const parent = nodeMap.get(d.parent.data.id)!
        node.parent = parent
        parent.children.push(node)
      }
    })

    const links: LayoutLink[] = (root as d3.HierarchyNode<D3TreeNode>).links().map((l) => ({
      source: nodeMap.get(l.source.data.id)!,
      target: nodeMap.get(l.target.data.id)!,
    }))

    return { nodes, links, width: containerWidth, height: containerHeight }
  }, [treeData, containerWidth, containerHeight])
}
