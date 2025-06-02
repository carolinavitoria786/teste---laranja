// script.js

// Mapeamento de quais pastas em 'web_models' correspondem a cada modelo + suas labels
const modelMap = {
  catdog: {
    path: 'web_models/catdog/model.json',
    labels: ['cat', 'dog']
  },
  orange: {
    path: 'web_models/orange/model.json',
    labels: ['fresh_orange', 'rotten_orange', 'sweet_orange']
  }
};

let currentModel = null;        // Variável global para armazenar o modelo carregado
const modelSelect = document.getElementById('modelSelect');
const imageInput  = document.getElementById('imageInput');
const previewImg  = document.getElementById('preview');
const resultDiv   = document.getElementById('result');

/**
 * Função que carrega o modelo TensorFlow.js escolhido no <select>.
 * Esta função é chamada sempre que o usuário muda a opção de modelo,
 * ou quando a página é carregada (para carregar o modelo padrão).
 */
async function loadModel() {
  const modelId = modelSelect.value;
  const info = modelMap[modelId];

  resultDiv.textContent = '🔄 Carregando modelo...';
  // Carrega o modelo a partir de info.path
  currentModel = await tf.loadLayersModel(info.path);
  resultDiv.textContent = '✅ Modelo carregado! Agora envie uma imagem.';
}

// Ao carregar a página, já carregamos o modelo padrão (valor inicial do <select>)
window.addEventListener('load', loadModel);

// Quando o usuário trocar a opção de modelo, recarrega o novo modelo
modelSelect.addEventListener('change', loadModel);


/**
 * Função auxiliar que recebe uma imagem HTMLImageElement,
 * converte para tensor, faz o resize e normaliza para [0,1].
 * Retorna um tensor de forma [1, inputSize, inputSize, 3].
 */
function preprocessImage(imgElement, inputSize) {
  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(imgElement)
      .resizeBilinear([inputSize, inputSize])
      .toFloat()
      .div(255.0)
      .expandDims();
    return tensor;
  });
}

/**
 * Função principal executada quando o usuário seleciona um arquivo.
 * - Exibe preview da imagem.
 * - Pré-processa a imagem para o modelo carregado.
 * - Executa a predição e mostra o resultado.
 */
imageInput.addEventListener('change', async () => {
  const file = imageInput.files[0];
  if (!file || !currentModel) {
    return;
  }

  // 1) Mostrar preview da imagem
  const objectURL = URL.createObjectURL(file);
  previewImg.src = objectURL;
  previewImg.hidden = false;

  // Espera até a imagem carregar completamente no <img>
  await new Promise((resolve) => {
    previewImg.onload = () => resolve();
  });

  // 2) Pré-processar para as dimensões de entrada do modelo
  const inputSize = currentModel.inputs[0].shape[1]; // assume [null, height, width, 3]
  const inputTensor = preprocessImage(previewImg, inputSize);

  // 3) Executar predição
  const prediction = currentModel.predict(inputTensor);
  const data = await prediction.data();   // obtém array de probabilidades
  prediction.dispose();
  inputTensor.dispose();

  // 4) Encontrar índice do maior valor
  const labels = modelMap[modelSelect.value].labels;
  let maxIdx = 0;
  for (let i = 1; i < data.length; i++) {
    if (data[i] > data[maxIdx]) {
      maxIdx = i;
    }
  }
  const confidence = (data[maxIdx] * 100).toFixed(1);

  // 5) Mostrar resultado
  resultDiv.textContent = `Resultado: ${labels[maxIdx]} (${confidence}%)`;
});
