import type { ReactNode } from 'react'
import Sidebar from './Sidebar'

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      {/* Main content — on mobile add top padding for the fixed header */}
      <main className="flex-1 overflow-y-auto p-4 md:p-6 pt-16 md:pt-6">
        {children}
      </main>
    </div>
  )
}
