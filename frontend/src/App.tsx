import React, { useEffect, useState } from 'react'
import { ControlPanel } from './components/ControlPanel/ControlPanel'
import { Navigator } from './components/Navigator/Navigator'
import { TreeVisualizer } from './components/TreeVisualizer/TreeVisualizer'
import { QueryFlow } from './components/QueryFlow/QueryFlow'
import { scenesApi } from './api/client'
import { useSceneStore } from './store/sceneStore'

const DEFAULT_PLY = '/scenes/ConferenceHall.ply'

type Tab = 'navigator' | 'tree' | 'query'

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('navigator')
  const [plyUrl, setPlyUrl] = useState<string>(DEFAULT_PLY)
  const { sceneId, setSceneInfo } = useSceneStore()

  useEffect(() => {
    scenesApi.init('default')
      .then(() => scenesApi.get('default'))
      .then((response) => setSceneInfo(response.data))
      .catch(() => {
        /* backend may still be starting */
      })
  }, [setSceneInfo])

  const tabs: { id: Tab; label: string }[] = [
    { id: 'navigator', label: '🧭 Navigator' },
    { id: 'tree',      label: '🌳 Semantic Tree' },
    { id: 'query',     label: '🔍 Query Flow' },
  ]

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white overflow-hidden">
      {/* ── Top bar ─────────────────────────────────────────────────── */}
      <header className="flex items-center gap-0 bg-gray-900 border-b border-gray-800 px-4 h-10 flex-shrink-0">
        <span className="text-sm font-bold text-green-400 mr-4">SemanticSplat</span>
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`px-3 h-10 text-xs font-medium transition-colors border-b-2 ${
              activeTab === t.id
                ? 'border-green-500 text-white'
                : 'border-transparent text-gray-400 hover:text-gray-200'
            }`}
          >
            {t.label}
          </button>
        ))}
      </header>

      {/* ── Control Panel ────────────────────────────────────────────── */}
      <ControlPanel
        plyUrl={plyUrl}
        onPlyUrlChange={setPlyUrl}
        onQuerySubmitted={() => setActiveTab('query')}
      />

      {/* ── Main content ─────────────────────────────────────────────── */}
      <main className="flex-1 min-h-0 overflow-hidden">
        {activeTab === 'navigator' && (
          <Navigator plyUrl={plyUrl || null} showSidebar />
        )}

        {activeTab === 'tree' && (
          <div className="h-full flex items-start justify-center p-4 overflow-auto">
            <TreeVisualizer width={Math.min(1200, window.innerWidth - 48)} height={600} />
          </div>
        )}

        {/* Keep mounted so polling + session state survive tab switches */}
        <div className={activeTab === 'query' ? 'h-full p-4' : 'hidden'}>
          <QueryFlow sceneId={sceneId} />
        </div>
      </main>
    </div>
  )
}
