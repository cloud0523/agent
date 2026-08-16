import { useState, useEffect } from 'react';
import { apiGet, apiPost, apiDelete } from '../api/client';

export default function DocumentPanel({ onSelectDoc }) {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [deleting, setDeleting] = useState(null);
  const [replacing, setReplacing] = useState(null);
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

  async function handleReplace(docId, filename, e) {
    e.stopPropagation(); // 不触发选中
    const file = e.target.files?.[0];
    if (!file) return;
    if (!window.confirm(`确认用 "${file.name}" 替换 "${filename}"？`)) {
      e.target.value = '';
      return;
    }

    setReplacing(docId);
    setError('');
    const formData = new FormData();
    formData.append('file', file);

    try {
      await apiPost(`/api/documents/${docId}/reingest/upload`, formData, true);
      e.target.value = '';
      await loadDocuments();
    } catch (err) {
      setError(err.message);
    } finally {
      setReplacing(null);
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
            className={`doc-list__item ${deleting === doc.id ? 'doc-list__item--deleting' : ''}`}
            onClick={() => onSelectDoc?.(doc.id)}
          >
            <div className="doc-list__primary">{doc.filename}</div>
            <div className="doc-list__secondary">
              <span>{doc.num_chunks} chunks</span>
              <span>{doc.file_type || 'unknown'}</span>
            </div>
            <label
              className="doc-list__replace-btn"
              title="替换文档"
              onClick={(e) => e.stopPropagation()}
            >
              {replacing === doc.id ? '...' : '⇄'}
              <input
                type="file"
                accept=".pdf,.docx,.txt,.md"
                style={{ display: 'none' }}
                disabled={replacing === doc.id}
                onChange={(e) => handleReplace(doc.id, doc.filename, e)}
              />
            </label>
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
