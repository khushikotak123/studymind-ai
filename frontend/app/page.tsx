'use client'
import { useState } from 'react'
import axios from 'axios'

const API = '/api'

export default function Home() {
  const [indexName, setIndexName] = useState('')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [topic, setTopic] = useState('')
  const [quiz, setQuiz] = useState<any[]>([])
  const [selected, setSelected] = useState<{[k:number]:string}>({})
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [activeTab, setActiveTab] = useState<'chat'|'quiz'>('chat')

  async function uploadPDF(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    const form = new FormData()
    form.append('file', file)
    const res = await axios.post(`${API}/upload-pdf`, form)
    setIndexName(res.data.index_name)
    setUploading(false)
    alert(`PDF ready! Index: ${res.data.index_name}`)
  }

  async function askQuestion() {
    if (!indexName || !question) return
    setLoading(true)
    setAnswer('')
    const res = await axios.post(`${API}/ask`, { index_name: indexName, question })
    setAnswer(res.data.answer)
    setLoading(false)
  }

  async function generateQuiz() {
    if (!indexName || !topic) return
    setLoading(true)
    setQuiz([])
    setSelected({})
    const res = await axios.post(`${API}/generate-quiz`, {
      index_name: indexName,
      topic,
      num_questions: 3
    })
    setQuiz(res.data.questions)
    setLoading(false)
  }

  return (
    <main className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-2xl mx-auto">

        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-900">StudyMind AI</h1>
          <p className="text-gray-500 mt-1">Upload your notes. Ask questions. Take quizzes.</p>
        </div>

        {/* Upload */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 mb-5">
          <p className="text-sm font-medium text-gray-700 mb-2">Upload your PDF notes</p>
          <input type="file" accept=".pdf" onChange={uploadPDF}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"/>
          {uploading && <p className="text-blue-500 text-sm mt-2">Processing PDF...</p>}
          {indexName && <p className="text-green-600 text-sm mt-2">Ready: {indexName}</p>}
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-4">
          <button onClick={() => setActiveTab('chat')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeTab === 'chat' ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 border border-gray-200'}`}>
            Chat with notes
          </button>
          <button onClick={() => setActiveTab('quiz')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeTab === 'quiz' ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 border border-gray-200'}`}>
            Take a quiz
          </button>
        </div>

        {/* Chat Tab */}
        {activeTab === 'chat' && (
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <textarea
              value={question}
              onChange={e => setQuestion(e.target.value)}
              placeholder="Ask anything from your notes..."
              className="w-full border border-gray-200 rounded-lg p-3 text-sm resize-none h-24 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button onClick={askQuestion} disabled={loading || !indexName}
              className="mt-3 w-full bg-blue-600 text-white py-2 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
              {loading ? 'Thinking...' : 'Ask StudyMind AI'}
            </button>
            {answer && (
              <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                <p className="text-sm font-medium text-blue-800 mb-1">Answer</p>
                <p className="text-sm text-gray-700">{answer}</p>
              </div>
            )}
          </div>
        )}

        {/* Quiz Tab */}
        {activeTab === 'quiz' && (
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <input
              value={topic}
              onChange={e => setTopic(e.target.value)}
              placeholder="Enter topic e.g. PHP, Web Servers..."
              className="w-full border border-gray-200 rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button onClick={generateQuiz} disabled={loading || !indexName}
              className="mt-3 w-full bg-blue-600 text-white py-2 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
              {loading ? 'Generating...' : 'Generate Quiz'}
            </button>

            {quiz.length > 0 && (
              <div className="mt-5 space-y-5">
                {quiz.map((q, i) => (
                  <div key={i} className="border border-gray-100 rounded-lg p-4">
                    <p className="text-sm font-medium text-gray-800 mb-3">Q{i+1}: {q.question}</p>
                    <div className="space-y-2">
                      {q.options.map((opt: string) => {
                        const isSelected = selected[i] === opt
                        const isCorrect = opt === q.answer
                        const showResult = selected[i] !== undefined
                        let style = 'border-gray-200 text-gray-700'
                        if (showResult && isCorrect) style = 'border-green-400 bg-green-50 text-green-800'
                        else if (showResult && isSelected) style = 'border-red-400 bg-red-50 text-red-800'
                        return (
                          <button key={opt} onClick={() => setSelected(s => ({...s, [i]: opt}))}
                            className={`w-full text-left text-sm px-3 py-2 rounded-lg border transition-colors ${style}`}>
                            {opt}
                          </button>
                        )
                      })}
                    </div>
                    {selected[i] && (
                      <p className="text-xs text-gray-500 mt-2">{q.explanation}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  )
}