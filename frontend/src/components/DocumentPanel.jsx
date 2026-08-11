import { useState, useEffect } from 'react';
import { apiGet, apiPost, apiDelete } from '../api/client';

export default function DocumentPanel({ onSelectDoc }) {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [deleting, setDeleting] = useState(null);
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

  async function handleDelete(docId, filename, e) {
    e.stopPropagation(); // 不触发选中
    if (!window.confirm(`确认删除 "${filename}"？`)) return;

    setDeleting(docId);
    setError('');

    try {
      await apiDelete(`/api/documents/${docId}`);
      await loadDocuments();
    } catch (err) {
      setError(err.message);
    } finally {
      setDeleting(null);
    }
  }

  async function handleUpload(e) {

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
            className={`doc-list__item ${deleting === doc.id ? 'doc-list__item--deleting' : ''}`}
            onClick={() => onSelectDoc?.(doc.id)}
          >
            <div className="doc-list__primary">{doc.filename}</div>
            <div className="doc-list__secondary">
              <span>{doc.chunk_count} chunks</span>
              <span>{doc.file_type || 'unknown'}</span>
            </div>
            <button
              className="doc-list__delete-btn"
              title="删除文档"
              disabled={deleting === doc.id}
              onClick={(e) => handleDelete(doc.id, doc.filename, e)}
            >
              {deleting === doc.id ? '...' : '×'}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
