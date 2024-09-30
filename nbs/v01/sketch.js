const URL = 'https://teachablemachine.withgoogle.com/models/lqYKlVl6-/';
let model, webcam, ctx, labelContainer, maxPredictions;

async function init() {
    const modelURL = URL + 'model.json';
    const metadataURL = URL + 'metadata.json';

    // Cargar el modelo y los metadatos
    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

    // Configurar la webcam
    const flip = true; 
    webcam = new tmPose.Webcam(640, 520, flip);
    await webcam.setup();
    webcam.play();
    window.requestAnimationFrame(loop);

    // Agregar elementos al DOM
    const canvas = document.getElementById('canvas');
    canvas.width = 640; 
    canvas.height = 520;
    ctx = canvas.getContext('2d');
    labelContainer = document.getElementById('label-container');
    for (let i = 0; i < maxPredictions; i++) {
        labelContainer.appendChild(document.createElement('div'));
    }
}

async function loop(timestamp) {
    webcam.update(); 
    await predict();
    window.requestAnimationFrame(loop);
}

async function predict() {
    // Predicción #1: ejecutar entrada a través de posenet
    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
    // Predicción 2: ejecutar entrada a través del modelo de clasificación de Teachable Machine
    const prediction = await model.predict(posenetOutput);

    let bestLabel = "";
    let highestProbability = 0;

    for (let i = 0; i < maxPredictions; i++) {
        const probability = prediction[i].probability.toFixed(2);
        const className = prediction[i].className;
        labelContainer.childNodes[i].innerHTML = `${className}: ${probability}`;

        // Determinar la etiqueta con mayor probabilidad
        if (probability > highestProbability) {
            highestProbability = probability;
            bestLabel = className;
        }
    }

    // Dibujar los resultados en el canvas
    drawPose(pose, bestLabel);
}

function drawPose(pose, bestLabel) {
    ctx.drawImage(webcam.canvas, 0, 0);

    // Dibujar los keypoints y el esqueleto
    if (pose) {
        const minPartConfidence = 0.5;
        tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
        tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);

        // Mostrar el resultado (Buena Postura / Mala Postura) en el canvas
        ctx.font = "40px Arial";
        ctx.fillStyle = "white";
        ctx.textAlign = "center";
        ctx.fillText(bestLabel, canvas.width / 2, 50);
    }
}
