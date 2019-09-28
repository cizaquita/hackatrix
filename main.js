// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

// Number of classes to classify
const NUM_CLASSES = 5;
// Webcam Image size. Must be 227. 
const IMAGE_SIZE = 400;
// K value for KNN
const TOPK = 10;


class Main {
  constructor() {

    // integración firebase
    /*var firebase = require('firebase');
    var request = require('request');

    var API_KEY = "AAAAx7N76uw:APA91bGsNaZKCXoR_cSqIwZZ9HYE72mFlj24_GHOdS3mofcgtLrox6orFHgP0_ti5boyYXWQQijXqs9fDJHDZYUo6luoIfdD8zm2iJaJfBLRkEmD_xDHZhGp8-2-L9U5NJfBmH96MmFh";

    firebase.initializeApp({
      serviceAccount: "account.json",
      databaseURL: "FIREBASE_DATABASE_URL"
    });
    ref = firebase.database().ref();
*/

    // Initiate variables
    this.infoTexts = [];
    this.logPanel = document.createElement('p');
    this.training = -1; // -1 when no class is being trained
    this.videoPlaying = false;

    // Initiate deeplearn.js math and knn classifier objects
    this.bindPage();

    // Create video element that will contain the webcam image
    this.video = document.createElement('video');
    this.video.setAttribute('autoplay', '');
    this.video.setAttribute('playsinline', '');

    // Add video element to DOM
    document.body.appendChild(this.video);

    // Create training buttons and info texts    
    for (let i = 0; i < NUM_CLASSES; i++) {
      const div = document.createElement('div');
      document.body.appendChild(div);
      div.style.marginBottom = '10px';

      // Create training button
      const button = document.createElement('button')

      // Crear botones de entrenamiento personalizaos
      // 
      const indicadorBoton = i+1;
      if (i == 0)
        button.innerText = "(" + indicadorBoton + ") Necesidad Fisiológica";
      else if (i == 1)
        button.innerText = "(" + indicadorBoton + ") Seguridad Física";
      else if (i == 2)
        button.innerText = "(" + indicadorBoton + ") Llamada a Familiares";
      else if (i == 3)
        button.innerText = "(" + indicadorBoton + ") Asistencia Médica";
      else
        button.innerText = "Entrenar Posición normal del paciente.";

      button.className = "btn-info";


      div.appendChild(button);

      // Listen for mouse events when clicking the button
      button.addEventListener('mousedown', () => this.training = i);
      button.addEventListener('mouseup', () => this.training = -1);

      // Create info text
      const infoText = document.createElement('span')
      infoText.innerText = " No hay entrenamiento.";
      div.appendChild(infoText);
      this.infoTexts.push(infoText);

    }

    // LOG PANEL
    const divLog = document.createElement('div');
    this.logPanel = document.createElement('p');
    document.body.appendChild(divLog);
    divLog.style.marginBottom = '10px';

    divLog.appendChild(this.logPanel);

    // Setup webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = IMAGE_SIZE;
        this.video.height = IMAGE_SIZE;

        this.video.addEventListener('playing', () => this.videoPlaying = true);
        this.video.addEventListener('paused', () => this.videoPlaying = false);
      })

    // Create LOG panel
  }

  async bindPage() {
    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenetModule.load();

    this.start();
  }

  start() {
    if (this.timer) {
      this.stop();
    }
    this.video.play();
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  stop() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }


/*
  sendNotificationToUser(userId, message, onSuccess) {
    request({
      url: 'https://fcm.googleapis.com/fcm/send',
      method: 'POST',
      headers: {
        'Content-Type': ' application/json',
        'Authorization': 'key=' + API_KEY
      },
      body: JSON.stringify({
        notification: {
          title: message
        },
        to: '/topics/user_' + userId
      })
    }, function(error, response, body) {
      if (response.statusCode >= 400) {
        console.error('HTTP Error: ' + response.statusCode + ' - ' + response.statusMessage);
      } else {
        onSuccess();
      }
    });
  }
*/

  async animate() {
    if (this.videoPlaying) {
      // Get image data from video element
      const image = tf.fromPixels(this.video);

      let logits;
      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');

      // Train class if one of the buttons is held down
      if (this.training != -1) {
        logits = infer();

        // Add current image to classifier
        this.knn.addExample(logits, this.training)
      }

      const numClasses = this.knn.getNumClasses();
      if (numClasses > 0) {

        // If classes have been added run predict
        logits = infer();
        const res = await this.knn.predictClass(logits, TOPK);

        for (let i = 0; i < NUM_CLASSES; i++) {

          // The number of examples for each class
          const exampleCount = this.knn.getClassExampleCount();

          // Make the predicted class bold
          if (res.classIndex == i) {
            this.infoTexts[i].style.fontWeight = 'bold';
          } else {
            this.infoTexts[i].style.fontWeight = 'normal';
          }

          // Update info text
          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} entrenamientos cargados - ${res.confidences[i] * 100}% `;
          }
          // Do something bro, cuando haya coincidencia
          if (res.confidences[i] * 100 > 98) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} entrenamientos cargados - ${res.confidences[i] * 100}% - COINCIDENCIA! `;
            
            if (i == 0){
              //this.logPanel.innerText = "Necesidad Fisiológica registrada.";
              setTimeout(function(){
                //this.logPanel.innerText = "";
                //console.log("Necesidad Fisiológica registrada.")
                this.enviarNotificacion("Necesidad Fisiológica registrada.");
              }, 3000);
            }
            else if (i == 1){

              //this.logPanel.innerText = "Seguridad Física registrada.";
              setTimeout(function(){
                //this.logPanel.innerText = "";
                this.enviarNotificacion("Seguridad Física registrada.");
              }, 3000);
            }
            else if (i == 2){
              //this.logPanel.innerText = "Llamada a Familiares registrada.";
              setTimeout(function(){
                //this.logPanel.innerText = "";
                this.enviarNotificacion("Llamada a Familiares registrada.");
              }, 3000);
            }
            else if (i == 3){
              //this.logPanel.innerText = "Asistencia Médica registrada.";
              setTimeout(function(){
                //this.logPanel.innerText = "Asistencia Médica registrada.";
                this.enviarNotificacion("Asistencia Médica registrada.");
              }, 3000);
            }
            
            //console.log('Se ha encontrado una coincidencia!');
          }
          // Acción botón uno
          /*if (res.confidences[i] * 100 > 90) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} entrenamientos cargados - ${res.confidences[i] * 100}!`
            //console.log('Se ha encontrado una coincidencia!');
          }*/

          // acciones de la clase en un determinado tiempo
          /*while (res.confidences[0] * 100 > 90){
            setTimeout(function(){
              alert("hey"); 
            }, 3000);
          }*/
        }
      }
      // Dispose image when done
      image.dispose();
      if (logits != null) {
        logits.dispose();
      }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }


}

window.addEventListener('load', () => new Main());