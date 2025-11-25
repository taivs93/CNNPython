export async function predictMnist(file) {
  const formData = new FormData();
  formData.append('file', file);
  const resp = await fetch('http://localhost:5000/api/predict/mnist', {
    method: 'POST',
    body: formData
  });
  return resp.json();
}

export async function predictShapes(file) {
  const formData = new FormData();
  formData.append('file', file);
  const resp = await fetch('http://localhost:5000/api/predict/shapes', {
    method: 'POST',
    body: formData
  });
  return resp.json();
}