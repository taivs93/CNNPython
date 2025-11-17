export async function predictMnist(base64) {
  const resp = await fetch('http://localhost:5000/predict/mnist', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ image: base64 })
  });
  return resp.json();
}
export async function predictShapes(base64) {
  const resp = await fetch('http://localhost:5000/predict/shapes', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ image: base64 })
  });
  return resp.json();
}