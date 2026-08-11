import { useState, useEffect } from 'react';
import { apiGet, apiPost } from '../api/client';

export default function DocumentPanel({ onSelectDoc }) {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    loadDocuments();
  }, []);

  async function loadDocuments() {
    try {
      const docs = await apiGet('/api/documents');
      setDocuments(docs);
    } catch (e) {
      setError(e.message);
    }
  }

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError('');
    const formData = new FormData();
    formData.append('file', file);

    try {
      await apiPost('/api/ingest/upload', formData, true);
      e.target.value = '';
      await loadDocuments();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="document-panel">
      <div className="document-panel__header">
        <h2>📄 文档</h2>
        <div className="document-panel__upload">
          <label className="upload-button">
            选择文件
            <input
              type="file"
              accept=".pdf,.docx,.txt,.md"
              onChange={handleUpload}
              disabled={uploading}
            />
          </label>
          {uploading && <span className="upload-status">上传中...</span>}
        </div>
      </div>

      {error && <div className="document-panel__error">{error}</div>}

      <ul className="doc-list">
        {documents.map((doc) => (
          <li
            key={doc.id}
            className="doc-list__item"
            onClick={() => onSelectDoc?.(doc.id)}
          >
            <div className="doc-list__primary">{doc.filename}</div>
            <div className="doc-list__secondary">
              <span>{doc.chunk_count} chunks</span>
              <span>{doc.file_type || 'unknown'}</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
