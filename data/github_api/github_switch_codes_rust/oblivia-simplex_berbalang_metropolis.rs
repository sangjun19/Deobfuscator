// Repository: oblivia-simplex/berbalang
// File: src/evolution/metropolis.rs

use std::sync::atomic::Ordering;

use rand::Rng;

use crate::configure::Config;
use crate::evolution::{Genome, Phenome};
use crate::observer::Observer;
use crate::ontogenesis::Develop;
use crate::util::random::hash_seed_rng;
use crate::EPOCH_COUNTER;

pub struct Metropolis<E: Develop<P>, P: Phenome + Genome + 'static> {
    pub specimen: P,
    pub config: Config,
    pub iteration: usize,
    pub observer: Observer<P>,
    pub evaluator: E,
    pub best: Option<P>,
}

impl<E: Develop<P>, P: Phenome + Genome + 'static> Metropolis<E, P> {
    pub fn new(config: &Config, observer: Observer<P>, evaluator: E) -> Self {
        let specimen = P::random(&config, 1);

        Self {
            specimen,
            config: config.clone(),
            iteration: 0,
            observer,
            evaluator,
            best: None,
        }
    }

    pub fn evolve(self) -> Self {
        let Self {
            specimen,
            config,
            iteration,
            observer,
            evaluator,
            best,
        } = self;

        EPOCH_COUNTER.fetch_add(1, Ordering::Relaxed);

        let mut specimen = if specimen.fitness().is_none() {
            evaluator.develop(specimen)
        } else {
            specimen
        };
        let variation = Genome::mate(&[&specimen, &specimen], &config);
        let variation = evaluator.develop(variation);

        let mut rng = hash_seed_rng(&specimen);
        let vari_fit = variation.scalar_fitness(&config.fitness.weighting).unwrap();
        let spec_fit = specimen.scalar_fitness(&config.fitness.weighting).unwrap();
        let delta = if (vari_fit - spec_fit).abs() < std::f64::EPSILON {
            0.0
        } else {
            1.0 / (vari_fit - spec_fit)
        };

        // if the variation is fitter, replace.
        // otherwise, let there be a chance of replacement inversely proportionate to the
        // difference in fitness.
        if delta < 0.0 || ((-delta).exp()) < rng.gen_range(0.0, 1.0) {
            //if delta < 0.0 { // pure hillclimbing
            specimen = variation;
            log::info!(
                "[{}] best: {:?}. specimen: {}, variation: {} (delta {}), switching",
                iteration,
                best.as_ref()
                    .and_then(|p| p.scalar_fitness(&config.fitness.weighting)),
                spec_fit,
                vari_fit,
                delta
            );
        };
        observer.observe(specimen.clone());

        let mut updated_best = false;
        let best = match best {
            Some(b)
                if specimen.scalar_fitness(&config.fitness.weighting).unwrap()
                    < b.scalar_fitness(&config.fitness.weighting).unwrap() =>
            {
                updated_best = true;
                Some(specimen.clone())
            }
            None => {
                updated_best = true;
                Some(specimen.clone())
            }
            _ => best,
        };
        if updated_best {
            log::info!("new best: {:?}", best.as_ref().unwrap());
        }
        Self {
            specimen,
            config,
            iteration: iteration + 1,
            observer,
            evaluator,
            best,
        }
    }
}
