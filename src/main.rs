/// Thank you to Artem Kirsanov for making a great video on this topic: https://www.youtube.com/watch?v=SmZmBKc7Lrs

use rand::prelude::*;

type Flt = f32;
type Loss = Flt;
type Prediction = Flt;
type Predictions = Vec<(Prediction, Loss)>;

struct DataPoint {
    x: Flt,
    y: Flt,
}

impl DataPoint {
    fn new(x: Flt, y: Flt) -> Self {
        Self { x, y }
    }
}

type Dataset = Vec<DataPoint>;

struct Network {
    k0: Flt,
    k1: Flt,
}

impl Network {
    fn new(rng: &mut StdRng) -> Self {
        let range = 10 as Flt;
        let mut gen = || (rng.gen::<Flt>() - 0.5 as Flt) * range;
        Self {
            k0: gen(),
            k1: gen(),
        }
    }

    fn predict(&self, data_point: &DataPoint) -> Prediction {
        self.k0 + data_point.x * self.k1
    }

    fn loss(&self, data_point: &DataPoint, prediction: Prediction) -> Loss {
        (data_point.y - prediction).powf(2 as Flt)
    }

    fn forward(&self, dataset: &Dataset) -> Predictions {
        dataset.iter()
            .map(|p| {
                let y_pred = self.predict(p);
                let loss = self.loss(p, y_pred);
                (y_pred, loss)
            })
            .collect()
    }

    fn backward(&mut self, lr: Flt, dataset: &Dataset, predictions: Predictions) {
        let mut k0_ds = 0 as Flt;
        let mut k1_ds = 0 as Flt;
        for ((pred, loss), pt) in predictions.iter().zip(dataset) {
            k0_ds += -2.0 * (pt.y - pred);
            k1_ds += -2.0 * (pt.y - pred) * pt.x;
        }
        let n = dataset.len() as Flt;
        self.k0 -= k0_ds * lr * (1.0 / n);
        self.k1 -= k1_ds * lr * (1.0 / n);
    }

    fn train(&mut self, learning_rate: Flt, iters: usize, dataset: &Dataset) {
        for i in 0..iters {
            print!("k0: {:.2}\tk1: {:.2}\t({i}/{iters})", self.k0, self.k1);

            let preds = self.forward(&dataset);
            let loss = preds.iter().map(|(p, l)| l).sum::<Loss>();
            print!("\tLoss: {loss}");

            self.backward(learning_rate, &dataset, preds);
            println!("\t\tNew params: k0: {:.2}\tk1: {:.2}", self.k0, self.k1);
        }
    }
}

fn main() {
    let dataset = vec![
        DataPoint::new(0.0, 0.0),
        DataPoint::new(1.0, 1.0),
    ];

    let mut rng = StdRng::seed_from_u64(0);
    let mut net = Network::new(&mut rng);
    net.train(0.01, 1000, &dataset);
}
