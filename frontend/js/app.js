// app.js
// Archivo principal que orquesta la carga de los módulos frontend.
// Se ejecuta una sola vez al cargar la página y se encarga de inyectar
// dinámicamente upload.js, metrics.js y dashboard.js.
// Así, si en el futuro quieres añadir o quitar módulos, basta
// con ajustar esta lista.

console.log('Demo Sales Forecasting cargado');

document.addEventListener('DOMContentLoaded', () => {
  const scripts = [
    '/static/js/upload.js',
    '/static/js/metrics.js',
    '/static/js/dashboard.js'
  ];

  // Función para cargar un <script> dinámicamente
  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src;
      s.async = false;       // respetar orden
      s.onload = () => {
        console.log(`Módulo cargado: ${src}`);
        resolve();
      };
      s.onerror = () => reject(new Error(`Error cargando ${src}`));
      document.body.appendChild(s);
    });
  }

  // Cargar todos los scripts en secuencia
  (async () => {
    try {
      for (const src of scripts) {
        await loadScript(src);
      }
      console.log('Todos los módulos frontend han sido cargados.');
    } catch (err) {
      console.error(err);
    }
  })();
});
