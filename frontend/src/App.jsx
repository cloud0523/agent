import { useState } from 'react'
import './App.css'
import DocumentPanel from './components/DocumentPanel'
import ChatPanel from './components/ChatPanel'

export default function App() {
  const [selectedDoc, setSelectedDoc] = useState(null)

  return (
    <div className="app">
      <aside className="sidebar">
        <DocumentPanel onSelectDoc={setSelectedDoc} />
      </aside>
      <main className="main">
        <ChatPanel selectedDoc={selectedDoc} />
      </main>
    </div>
  )
}
