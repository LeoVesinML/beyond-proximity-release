/**
 * TreeVisualizer — D3-powered interactive semantic tree visualisation.
 *
 * - Nodes coloured by type (root / zone / region / leaf)
 * - Visited / active nodes highlighted during query traversal
 * - Hover tooltip showing node summary
 * - Click a node to see its view_ids
 */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { treeApi } from '../../api/client'
import { useSceneStore } from '../../store/sceneStore'
import { useTreeStore } from '../../store/treeStore'
import { useQueryStore } from '../../store/queryStore'
import { useTreeLayout, type LayoutNode, type LayoutLink } from './useTreeLayout'
import type { D3TreeNode } from '../../types'

const NODE_COLORS: Record<string, string> = {
  root:   '#6366f1',   // indigo
  zone:   '#22c55e',   // green
  region: '#eab308',   // yellow
  leaf:   '#64748b',   // slate
}

const ACTIVE_COLOR  = '#f97316'   // orange
const VISITED_COLOR = '#7dd3fc'   // sky

interface TreeVisualizerProps {
  width?: number
  height?: number
}

export function TreeVisualizer({ width = 900, height = 500 }: TreeVisualizerProps) {
  const { sceneId } = useSceneStore()
  const { treeData, activeNodeId, setActiveNode, isLoading, setLoading, setTreeData } = useTreeStore()
  const { visitedNodeIds } = useQueryStore()

  const svgRef = useRef<SVGSVGElement>(null)
  const { nodes, links } = useTreeLayout(treeData, width - 20, height - 60)

  const [tooltip, setTooltip] = useState<{ x: number; y: number; node: LayoutNode } | null>(null)

  // Auto-load persisted tree whenever the active scene changes so users do not
  // need to manually click "Refresh Tree" after re-opening the app.
  useEffect(() => {
    let cancelled = false
    const loadTree = async () => {
      setLoading(true)
      try {
        const res = await treeApi.getViz(sceneId)
        if (!cancelled) setTreeData(res.data as D3TreeNode)
      } catch {
        if (!cancelled) setTreeData(null)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void loadTree()
    return () => { cancelled = true }
  }, [sceneId, setLoading, setTreeData])

  const nodeColor = useCallback(
    (n: LayoutNode) => {
      if (n.id === activeNodeId)          return ACTIVE_COLOR
      if (visitedNodeIds.includes(n.id))  return VISITED_COLOR
      return NODE_COLORS[n.type] ?? '#94a3b8'
    },
    [activeNodeId, visitedNodeIds],
  )

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return
    const svg = d3.select(svgRef.current)

    // Links
    const linkSel = svg.select<SVGGElement>('.links')
      .selectAll<SVGPathElement, LayoutLink>('path')
      .data(links, (d) => `${d.source.id}-${d.target.id}`)

    linkSel.join(
      (enter) => enter.append('path')
        .attr('fill', 'none')
        .attr('stroke', '#374151')
        .attr('stroke-width', 1.5),
    ).attr('d', (d) =>
      `M${d.source.x},${d.source.y} C${d.source.x},${(d.source.y + d.target.y) / 2} ${d.target.x},${(d.source.y + d.target.y) / 2} ${d.target.x},${d.target.y}`,
    )

    // Nodes
    const nodeSel = svg.select<SVGGElement>('.nodes')
      .selectAll<SVGCircleElement, LayoutNode>('circle')
      .data(nodes, (d) => d.id)

    nodeSel.join(
      (enter) => enter.append('circle')
        .attr('r', (d) => (d.type === 'root' ? 14 : d.type === 'zone' ? 10 : 7))
        .attr('stroke', '#111827')
        .attr('stroke-width', 1.5)
        .attr('cursor', 'pointer')
        .on('click', (_, d) => setActiveNode(d.id === activeNodeId ? null : d.id))
        .on('mouseover', (event, d) => setTooltip({ x: d.x, y: d.y, node: d }))
        .on('mouseout', () => setTooltip(null)),
    )
      .attr('cx', (d) => d.x)
      .attr('cy', (d) => d.y)
      .attr('fill', nodeColor)

    // Labels
    const labelSel = svg.select<SVGGElement>('.labels')
      .selectAll<SVGTextElement, LayoutNode>('text')
      .data(nodes, (d) => d.id)

    labelSel.join(
      (enter) => enter.append('text')
        .attr('text-anchor', 'middle')
        .attr('font-size', 10)
        .attr('fill', '#d1d5db')
        .attr('dy', '1.8em')
        .attr('pointer-events', 'none'),
    )
      .attr('x', (d) => d.x)
      .attr('y', (d) => d.y)
      .text((d) => d.name.length > 14 ? d.name.slice(0, 12) + '…' : d.name)
  }, [nodes, links, nodeColor, activeNodeId, setActiveNode])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        <span className="animate-spin mr-2">⟳</span> Loading tree…
      </div>
    )
  }

  if (!treeData) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-sm text-center p-4">
        No semantic tree yet.<br />
        Capture views, then ask Cursor AI to build the tree via MCP tools.
      </div>
    )
  }

  return (
    <div className="relative overflow-auto bg-gray-950 rounded-lg border border-gray-800" style={{ width, height }}>
      {/* Legend */}
      <div className="absolute top-2 left-2 flex gap-3 text-xs text-gray-400 bg-gray-900/80 rounded px-2 py-1">
        {Object.entries(NODE_COLORS).map(([type, color]) => (
          <span key={type} className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: color }} />
            {type}
          </span>
        ))}
        <span className="flex items-center gap-1">
          <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: VISITED_COLOR }} />
          visited
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: ACTIVE_COLOR }} />
          active
        </span>
      </div>

      <svg ref={svgRef} width={width} height={height}>
        <g className="links" />
        <g className="nodes" />
        <g className="labels" />
      </svg>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="absolute bg-gray-800 border border-gray-600 rounded shadow-lg p-2 text-xs text-gray-200 max-w-[220px] z-20 pointer-events-none"
          style={{ left: tooltip.x + 12, top: tooltip.y - 10 }}
        >
          <p className="font-semibold text-white">{tooltip.node.name}</p>
          <p className="text-gray-400 mt-0.5">{tooltip.node.type}</p>
          <p className="mt-1 text-gray-300 leading-tight">{tooltip.node.summary}</p>
          {tooltip.node.view_ids.length > 0 && (
            <p className="mt-1 text-green-400">{tooltip.node.view_ids.length} views</p>
          )}
        </div>
      )}
    </div>
  )
}
