import React, { useState, useEffect } from 'react';
import axios from 'axios';
import UploadForm from './components/UploadForm';
import Metrics from './components/Metrics';
import ForecastTable from './components/ForecastTable';

function App() {
  const [metrics, setMetrics] = useState(null);
  const [predictions, setPredictions] = useState([]);

  // Al montar, obtener métricas
  useEffect(() => {
    axios.get('http://localhost:8000/metrics')
      .then(res => setMetrics(res.data.metrics))
      .catch(err => console.error(err));
  }, []);

  const handleFileUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post('http://localhost:8000/predict_csv', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPredictions(res.data.predictions);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Demo de Sales Forecasting</h1>
      <section className="mb-6">
        <h2 className="text-xl font-semibold">Métricas del Modelo</h2>
        {metrics ? <Metrics metrics={metrics} /> : <p>Cargando métricas...</p>}
      </section>

      <section className="mb-6">
        <h2 className="text-xl font-semibold">Subir CSV para Predicción</h2>
        <UploadForm onUpload={handleFileUpload} />
      </section>

      <section>
        <h2 className="text-xl font-semibold">Resultados de Predicción</h2>
        <ForecastTable predictions={predictions} />
      </section>
    </div>
  );
}

export default App;