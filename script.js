// Training data
const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
const ys = tf.tensor2d([-3, -1, 2, 3, 5, 7], [6, 1]);

// Build model
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({
  loss: 'meanSquaredError',
  optimizer: 'sgd'
});

// Train the model
async function doTraining() {
  await model.fit(xs, ys, {
    epochs: 500,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log("Epoch:" + epoch + " Loss:" + logs.loss);
      }
    }
  });
  console.log("Training Complete!");
}

// Predict function
async function makePrediction() {
  const inputVal = parseFloat(document.getElementById("enterNumber").value);
  const outputBox = document.getElementById("outputBox");

  if (isNaN(inputVal)) {
    outputBox.innerText = "Please enter a valid number!";
    return;
  }

  const output = model.predict(tf.tensor2d([inputVal], [1, 1]));
  const prediction = output.dataSync()[0];
  outputBox.innerText = "Predicted output is: " + prediction.toFixed(2);
}

// Start training when page loads
window.onload = doTraining;
