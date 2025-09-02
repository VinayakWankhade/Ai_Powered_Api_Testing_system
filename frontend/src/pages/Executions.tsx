import { PlayIcon } from '@heroicons/react/24/outline'

export default function Executions() {
  return (
    <div className="p-6">
      <div className="text-center py-12">
        <PlayIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Test Executions</h1>
        <p className="text-gray-600">
          Monitor test execution sessions with real-time progress tracking and detailed results.
        </p>
      </div>
    </div>
  )
}
