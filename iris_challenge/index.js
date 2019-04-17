const tf = require('@tensorflow/tfjs');
const irisTraining = require('./training.json');
const irisTesting = require('./testing.json');


//creando los datos de entrenamiento    (Mapeo)

const trainingData = tf.tensor2d(irisTraining.map(item =>[
        item.sepal_length, 
        item.sepal_width, 
        item.petal_length, 
        item.petal_width
    ]
),[130, 4])

//creaanto los datos de testing     (Mapeo)

const testingData = tf.tensor2d(irisTesting.map(item =>[
        item.sepal_length, 
        item.sepal_width, 
        item.petal_length, 
        item.petal_width
    ]
),[14, 4])

// creando modelo

const outputData = tf.tensor2d(irisTraining.map( item => [
        item.species === 'setosa' ? 1 : 0,
        item.species === 'virginica' ? 1 : 0,
        item.species === 'versicolor' ? 1 : 0,
    ]
),[130, 3])

//creamos el modelo, que sera de tipo secuencial

const model = tf.sequential();

// creamos la primera capa, la cual nos ayudará a obtener los primeros resultados, estos seran 10
model.add(tf.layers.dense({
    inputShape: 4,     //cuatro entradas de datos
    activation: 'sigmoid',     //activamacion sigmoide 
    units: 10      //los resultados devueltos seran 10 unidades
}));

//segunda capa, que mediante los 10 datos obtenidos anteriormente, volveran a ser procesados, para obtener la prediccion correcta
model.add(tf.layers.dense({
    inputShape: 10,        //entradas de datos
    activation: 'softmax',     //activacion softmax, normaliza los valores, la cual suman en su total 1 (probabilidades)
    units: 3       //3 salidas de datos, las cuales nos diran a que clase de flor pertenece
}));

model.summary();    //chequearlo

// compilando el modelo

model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam()
})

//creando el algoritmo que nos mostrara en pantalla, cada iteracion, mostrando su perdida durante la historia (cambio)

async function train_data(){
    console.log("***Perdida***")
    for(let i=0;i<40;i++){
        let res = await model.fit(trainingData,
            outputData,
            {epochs: 40});
        console.log(`${i}.: ${res.history.loss[0]}`);
    }
}


//creando la funcion principal, esta sera ejecutada por node

async function main(){
    await train_data();
    console.log('Modelo de prediccion');
    model.predict(testingData).print();
}

main();

