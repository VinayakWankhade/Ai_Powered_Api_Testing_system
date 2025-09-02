import { ChartBarIcon } from '@heroicons/react/24/outline'

export default function Coverage() {
  return (
    <div className="p-6">
      <div className="text-center py-12">
        <ChartBarIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Coverage Analytics</h1>
        <p className="text-gray-600">
          Comprehensive coverage reports with charts, trends, and detailed analytics for your API testing.
        </p>
      </div>
    </div>
  )
}
