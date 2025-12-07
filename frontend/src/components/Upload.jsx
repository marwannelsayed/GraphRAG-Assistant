import { useState } from 'react';

function Upload({ onUploadSuccess }) {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
  };

  const handleUpload = async () => {
    // TODO: Implement file upload logic
    // - Create FormData with selected files
    // - POST to /api/ingest/documents
    // - Handle upload progress
    // - Call onUploadSuccess callback
    setUploading(true);
    console.log('Upload files:', files);
    // Simulate upload
    setTimeout(() => {
      setUploading(false);
      setFiles([]);
      onUploadSuccess?.();
    }, 1000);
  };

  return (
    <div className="upload-container">
      <h2>Upload Documents</h2>
      <input
        type="file"
        multiple
        onChange={handleFileChange}
        disabled={uploading}
      />
      {files.length > 0 && (
        <div>
          <p>Selected files: {files.length}</p>
          <button onClick={handleUpload} disabled={uploading}>
            {uploading ? 'Uploading...' : 'Upload'}
          </button>
        </div>
      )}
    </div>
  );
}

export default Upload;

