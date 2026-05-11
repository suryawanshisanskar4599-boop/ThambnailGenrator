import React, { useState, useCallback, useEffect, useRef } from "react";
import "./App.css";

function App() {
  const [mediaList, setMediaList] = useState([]);
  const [currentMedia, setCurrentMedia] = useState(null);
  const [categories, setCategories] = useState({});
  const [selectedCategory, setSelectedCategory] = useState("");
  const [topThumbnails, setTopThumbnails] = useState([]);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [cache, setCache] = useState({});

  const intervalRef = useRef(null);
  const activeFileKeyRef = useRef("");
  const isProcessingRef = useRef(false);

  // ================= STOP POLLING =================
  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => () => stopPolling(), [stopPolling]);

  // ================= PROGRESS POLLING =================
  const pollProgress = useCallback((jobId, fileKey) => {
    stopPolling();

    intervalRef.current = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:5000/progress/${jobId}`);
        if (!res.ok) {
          throw new Error(`Progress request failed with ${res.status}`);
        }

        const data = await res.json();

        setProgress(data.percent || 0);
        setStatus(data.status || "");

        if (data.done) {
          stopPolling();

          if (data.status?.includes("Error")) {
            setIsProcessing(false);
            isProcessingRef.current = false;
            return;
          }

          if (activeFileKeyRef.current !== fileKey) {
            stopPolling();
            return;
          }

          const formatted = {};
          Object.keys(data.categories || {}).forEach((cat) => {
            formatted[cat] = data.categories[cat].map(
              (img) => `data:image/jpeg;base64,${img}`
            );
          });

          const top = (data.top || []).map(
            (img) => `data:image/jpeg;base64,${img}`
          );

          setCategories(formatted);
          setTopThumbnails(top);

          if (Object.keys(formatted).length > 0) {
            setSelectedCategory(Object.keys(formatted)[0]);
          }

          setCache((prev) => ({
            ...prev,
            [fileKey]: { categories: formatted, top: top },
          }));

          setIsProcessing(false);
          isProcessingRef.current = false;
        }
      } catch (err) {
        console.error("Polling error:", err);
        setStatus(`Error: ${err.message}`);
        setIsProcessing(false);
        isProcessingRef.current = false;
        stopPolling();
      }
    }, 1000);
  }, [stopPolling]);

  // ================= PROCESS VIDEO =================
  const processVideo = useCallback(
    async (file) => {
      const fileKey = `${file.name}-${file.size}-${file.lastModified}`;

      // Check cache first
      if (cache[fileKey]) {
        const cached = cache[fileKey];
        setCategories(cached.categories);
        setTopThumbnails(cached.top);
        const categoryKeys = Object.keys(cached.categories);
        setSelectedCategory(categoryKeys[0] || "");
        setStatus("Loaded from cache");
        setProgress(100);
        setIsProcessing(false);
        isProcessingRef.current = false;
        return;
      }

      if (isProcessingRef.current) {
        setStatus("Error: Already processing another video.");
        return;
      }

      activeFileKeyRef.current = fileKey;
      isProcessingRef.current = true;
      setCategories({});
      setTopThumbnails([]);
      setSelectedCategory("");
      setIsProcessing(true);
      setStatus("Uploading... Please wait");
      setProgress(0);

      try {
        const initRes = await fetch("http://localhost:5000/init_upload", { method: "POST" });
        const initData = await initRes.json();
        
        if (!initRes.ok) {
          throw new Error(initData.error || "Failed to initialize upload");
        }
        
        const jobId = initData.job_id;
        const chunkSize = 25 * 1024 * 1024; // 25 MB chunks
        const totalChunks = Math.ceil(file.size / chunkSize);
        
        let chunkData = null;
        for (let i = 0; i < totalChunks; i++) {
          if (activeFileKeyRef.current !== fileKey) return; // aborted

          const start = i * chunkSize;
          const end = Math.min(start + chunkSize, file.size);
          const chunk = file.slice(start, end);

          const formData = new FormData();
          formData.append("file", chunk);
          formData.append("chunk_index", i);
          formData.append("total_chunks", totalChunks);
          formData.append("job_id", jobId);
          formData.append("filename", file.name);

          const chunkRes = await fetch("http://localhost:5000/upload_chunk", {
            method: "POST",
            body: formData
          });
          
          chunkData = await chunkRes.json();
          if (!chunkRes.ok) throw new Error(chunkData.error || "Chunk upload failed");

          const percent = Math.round(((i + 1) / totalChunks) * 100);
          setProgress(percent);
          setStatus(`Uploading... ${percent}%`);
        }

        if (chunkData && chunkData.complete) {
          pollProgress(jobId, fileKey);
        }
      } catch (err) {
        setStatus(
          err.name === "AbortError"
            ? "Upload timed out"
            : `Error: ${err.message}`
        );
        setIsProcessing(false);
        isProcessingRef.current = false;
      }
    },
    [cache, pollProgress]
  );

  // ================= DRAG & DROP =================
  const handleDrop = (event) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);

    files.forEach((file) => {
      const url = URL.createObjectURL(file);
      const isVideo = file.type.startsWith("video/") || /\.(mp4|mkv|avi|mov|wmv|flv|webm)$/i.test(file.name);

      const mediaObj = {
        name: file.name,
        url,
        type: isVideo ? "video" : "image",
        file: isVideo ? file : null,
        id: `${file.name}-${Date.now()}`,
      };

      setMediaList((prev) => [...prev, mediaObj]);

      setCurrentMedia((prev) => prev || mediaObj);

      if (isVideo) processVideo(file);
    });
  };

  const handleDragOver = (e) => e.preventDefault();

  // ================= UI =================
  return (
    <div className="container">
      {/* LEFT SIDE: Media Focus */}
      <div className="left-panel">
        <div
          className={`media-stage ${!currentMedia ? "empty-stage" : ""}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          {!currentMedia ? (
            <div className="drop-prompt">
              <p style={{ color: 'var(--gold-soft)', fontSize: '0.85rem', letterSpacing: '0.05em', fontWeight: '500' }}>
                Drop video here to analyze
              </p>
            </div>
          ) : currentMedia.type === "video" ? (
            <video src={currentMedia.url} controls className="media-player" key={currentMedia.id} />
          ) : (
            <img src={currentMedia.url} alt="preview" className="media-player" />
          )}
        </div>

        <div className="library-panel">
          <h3 className="section-title text-gold">Archive</h3>
          <div className="library-grid">
            {mediaList.map((item) => (
              <div
                key={item.id}
                className={`library-item ${currentMedia?.id === item.id ? "active" : ""}`}
                onClick={() => {
                  setCurrentMedia(item);
                  if (item.type === "video") processVideo(item.file);
                }}
              >
                <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                  {item.type === "video" ? (
                    <video src={`${item.url}#t=0.1`} preload="metadata" style={{ width: '64px', height: '36px', objectFit: 'cover', borderRadius: '4px', backgroundColor: '#000' }} />
                  ) : (
                    <img src={item.url} alt="thumb" style={{ width: '64px', height: '36px', objectFit: 'cover', borderRadius: '4px', backgroundColor: '#000' }} />
                  )}
                  <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
                    <span style={{ fontSize: '0.65rem', color: 'var(--gold-soft)', letterSpacing: '0.05em', textTransform: 'uppercase' }}>FILE</span>
                    <p className="item-name" style={{ margin: '4px 0 0', fontSize: '0.8rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{item.name}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* RIGHT SIDE: Intelligence Panel */}
      <div className="right-panel">
        <h3 className="section-title text-gold">Analysis Engine</h3>

        {isProcessing && (
          <div className="progress-section">
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
              <span>{status}</span>
              <span style={{ color: 'var(--gold-soft)', fontWeight: '500' }}>{progress}%</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${progress}%`, background: 'var(--gold-primary)' }}></div>
            </div>
          </div>
        )}

        {!isProcessing && status.includes("Error") && (
          <div style={{ color: '#ff4444', fontSize: '0.8rem', padding: '12px', background: 'rgba(255, 0, 0, 0.1)', borderRadius: '6px', border: '1px solid #ff4444' }}>
            {status}
          </div>
        )}

        <div>
          <h3 className="section-title text-gold">Category</h3>
          <select
            className="modern-select"
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
          >
            {Object.keys(categories).length === 0 && <option>Waiting for data...</option>}
            {Object.keys(categories).map(cat => <option key={cat} value={cat}>{cat}</option>)}
          </select>
        </div>

        <div className="results-container">
          {categories[selectedCategory] && (
            <div style={{ marginBottom: '40px' }}>
              <h3 className="section-title text-gold">Captured Frames</h3>
              <div className="thumbnail-grid">
                {categories[selectedCategory].map((thumb, i) => (
                  <div key={i} className="thumb-card" onClick={() => setCurrentMedia({ type: 'image', url: thumb, id: `c-${i}` })}>
                    <img src={thumb} alt="thumb" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {topThumbnails.length > 0 && (
            <div>
              <h3 className="section-title text-gold">✦ Top Recommended</h3>
              <div className="thumbnail-grid">
                {topThumbnails.map((thumb, i) => (
                  <div key={i} className="thumb-card premium" style={{ position: 'relative' }} onClick={() => setCurrentMedia({ type: 'image', url: thumb, id: `t-${i}` })}>
                    <div style={{ position: 'absolute', top: '8px', left: '8px', background: 'var(--gold-primary)', color: '#000', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold', zIndex: 10 }}>Rank {i + 1}</div>
                    <img src={thumb} alt="top" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
