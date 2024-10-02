// El enlace a tu modelo proporcionado por el panel de exportación de Teachable Machine
const URL = "https://teachablemachine.withgoogle.com/models/lqYKlVl6-/";
let model, webcam, ctx, labelContainer, maxPredictions;

async function init() {
    const modelURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";

    // Cargar el modelo y los metadatos
    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

    // Función conveniente para configurar la webcam
    const size = 200;
    const flip = true; // Si se desea voltear la webcam
    webcam = new tmPose.Webcam(size, size, flip); // ancho, alto, voltear
    await webcam.setup(); // Solicitar acceso a la webcam
    await webcam.play();
    window.requestAnimationFrame(loop);

    // Obtener elementos del DOM
    const canvas = document.getElementById("canvas");
    canvas.width = size;
    canvas.height = size;
    ctx = canvas.getContext("2d");
    labelContainer = document.getElementById("label-container");
    for (let i = 0; i < maxPredictions; i++) { // Y etiquetas de clase
        labelContainer.appendChild(document.createElement("div"));
    }
}

async function loop(timestamp) {
    webcam.update(); // Actualizar el fotograma de la webcam
    await predict();
    window.requestAnimationFrame(loop);
}

async function predict() {
    // Predicción #1: Ejecutar la entrada a través de posenet
    // estimatePose puede tomar una imagen, video o elemento canvas html
    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
    // Predicción 2: Ejecutar la entrada a través del modelo de clasificación de Teachable Machine
    const prediction = await model.predict(posenetOutput);

    for (let i = 0; i < maxPredictions; i++) {
        const classPrediction =
            prediction[i].className + ": " + prediction[i].probability.toFixed(2);
        labelContainer.childNodes[i].innerHTML = classPrediction;
    }

    // Finalmente dibujar las posturas
    drawPose(pose);
}

function drawPose(pose) {
    if (webcam.canvas) {
        ctx.drawImage(webcam.canvas, 0, 0);
        // Dibujar los puntos clave y el esqueleto
        if (pose) {
            const minPartConfidence = 0.5;
            tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
            tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
        }
    }
}
