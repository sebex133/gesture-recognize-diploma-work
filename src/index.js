/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';
import * as tf from '@tensorflow/tfjs-core';

import * as tfjs from '@tensorflow/tfjs';
import * as tfnode from '@tensorflow/tfjs-node';

import * as mpHands from '@mediapipe/hands';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import * as handdetection from '@tensorflow-models/hand-pose-detection';

import {Camera} from './camera';
import {setupDatGui} from './option_panel';
import {STATE} from '../shared/params';
import {setupStats} from '../shared/stats_panel';
import {setBackendAndEnvFlags} from '../shared/util';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;

async function createDetector() {
  switch (STATE.model) {
    case handdetection.SupportedModels.MediaPipeHands:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return handdetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          maxHands: STATE.modelConfig.maxNumHands,
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${mpHands.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return handdetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          maxHands: STATE.modelConfig.maxNumHands
        });
      }
  }
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimateHandsStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimateHandsStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

var datasetProcess = false;
var datasetButton = document.getElementById("dataset-button");
var pointsDatasetArr = [];

datasetButton.addEventListener("click", function(e) {
    if (datasetProcess) {
      console.log("Stop dataset");
      e.target.innerHTML = "START DATASET SNAPSHOTING"
      datasetProcess = false;

      localStorage.setItem('pointsDatasetJson', JSON.stringify(pointsDatasetArr));
    }
    else {
      console.log("Start dataset");
      e.target.innerHTML = "SNAPPING DATASET... (click to stop)"
      datasetProcess = true;

      pointsDatasetArr = [];
    }
});

function processHandsData(hands) {
  // console.log(hands);
  if (hands[0] != undefined && hands[0].handedness != undefined) {
    pointsDatasetArr.push(hands[0].keypoints3D);

    // Wrist 2D info
    // console.log(hands[0].handedness + " " + hands[0].keypoints[0].name + " " + hands[0].keypoints[0].x + " " + hands[0].keypoints[0].y)
    // 3D info special points of hand

    // console.log("3d " + hands[0].handedness + " " + hands[0].keypoints3D[0].name + " " + hands[0].keypoints3D[0].x + " " + hands[0].keypoints3D[0].y+ " " + hands[0].keypoints3D[0].z)
    // console.log("3d " + hands[0].handedness + " " + hands[0].keypoints3D[5].name + " " + hands[0].keypoints3D[5].x + " " + hands[0].keypoints3D[5].y+ " " + hands[0].keypoints3D[5].z)
    // console.log("3d " + hands[0].handedness + " " + hands[0].keypoints3D[17].name + " " + hands[0].keypoints3D[17].x + " " + hands[0].keypoints3D[17].y+ " " + hands[0].keypoints3D[17].z)
    // console.log("3d " + hands[0].handedness + " " + hands[0].keypoints3D[12].name + " " + hands[0].keypoints3D[12].x + " " + hands[0].keypoints3D[12].y+ " " + hands[0].keypoints3D[12].z)
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let hands = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimateHands.
    beginEstimateHandsStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      hands = await detector.estimateHands(
        camera.video,
        {
          flipHorizontal: false
        }
      );

      // Collect data to dataset process.
      if (datasetProcess) {
        processHandsData(hands);
      }

      if (realtimePredictProcess) {
        predictGesture(hands);
      }

    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    endEstimateHandsStats();
  }

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (hands && hands.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(hands);
  }
}

async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = requestAnimationFrame(renderPrediction);
};

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);

  // Load the Tensorflow model.=
  const modelHandler = tfnode.io.fileSystem("./trained_model/model.json");
  const model = await tfjs.loadLayersModel(modelHandler);

  urlParams.append('model', 'mediapipe_hands');

  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }

  await setupDatGui(urlParams);

  stats = setupStats();

  camera = await Camera.setupCamera(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await createDetector();

  renderPrediction();
};

app();

function downloadFile(content, fileName, contentType) {
  var a = document.createElement("a");
  var file = new Blob([content], {type: contentType});
  a.href = URL.createObjectURL(file);
  a.download = fileName;
  a.click();
}

// Preapre and download JSON data for Tensor model
var modelNameTextbox = document.getElementById("model-name-textbox");
var modelName = '';
modelNameTextbox.addEventListener("change", function(e) {
  modelName = modelNameTextbox.value;
});

var modelButton = document.getElementById("model-button");
modelButton.addEventListener("click", function(e) {
  console.log("Start model");
  e.target.innerHTML = "PREPARING DATA FOR MODEL DOWNLOAD...";

  const points = JSON.parse(
      localStorage.getItem('pointsDatasetJson')
  );
  const handDataForTensor = [];

  for(let i = 0; i < points.length; i++) {
    let oneHandData = [];
    for(let j = 0; j < points[i].length; j++) {
      oneHandData.push([
        points[i][j].x,
        points[i][j].y,
        points[i][j].z
      ]);
    }
    handDataForTensor.push(oneHandData);
  }

  const jsonData = JSON.stringify(handDataForTensor);
  localStorage.setItem('tensorData', jsonData);

  console.log("Model name: " + modelName);
  if (modelName) {
    console.log("Downloading start...")
    downloadFile(jsonData, modelName + '.json', 'application/json');
    console.log("Downloading END")
  }

  e.target.innerHTML = "DOWNLOAD DATA FOR MODEL";
  console.log("Stop model");
});

// Realtime prediction
var realtimePredictProcess = false;
var realtimePredictButton = document.getElementById("dataset-button");

realtimePredictButton.addEventListener("click", function(e) {
    if (realtimePredictProcess) {
      e.target.innerHTML = "START PREDICTING"
      realtimePredictProcess = false;
    }
    else {
      e.target.innerHTML = "PREDICTING NOW... (click to stop)"
      realtimePredictProcess = true;
    }
});

function predictGesture(hands) {
  keypoints3d = hands[0].keypoints3D;
  const oneHandData = [];

  for(let i = 0; i < keypoints3d.length; i++) {
    oneHandData.push([
      points[i].x,
      points[i].y,
      points[i].z
    ]);
  }

  const prediction = model.predict([oneHandData]);
  console.log(prediction)
}