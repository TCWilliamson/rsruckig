//! `get_position_extrema` checked against the trajectory itself, sampled densely.
//!
//! A phase with zero jerk and non-zero acceleration whose velocity changes sign
//! turns at `t = -v / a`. Every reversal braked at the acceleration limit has such
//! a phase, so these tests drive plenty of them (and the second-order profiles,
//! which have zero jerk throughout).

use rsruckig::prelude::*;
use rsruckig::trajectory::Trajectory;

use rand_distr::{Distribution, Uniform};
use rand_pcg::Pcg64Mcg;

/// How far an extremum may lie beyond the sampled range.
const SAMPLE_TOL: f64 = 1e-6;

fn input_1dof(
    p0: f64,
    v0: f64,
    a0: f64,
    pf: f64,
    v_max: f64,
    a_max: f64,
    j_max: f64,
) -> InputParameter<0> {
    let mut input = InputParameter::<0>::new(Some(1));
    input.current_position[0] = p0;
    input.current_velocity[0] = v0;
    input.current_acceleration[0] = a0;
    input.target_position[0] = pf;
    input.max_velocity[0] = v_max;
    input.max_acceleration[0] = a_max;
    input.max_jerk[0] = j_max;
    input
}

fn plan(input: &InputParameter<0>) -> Trajectory<0> {
    let mut otg = Ruckig::<0, ThrowErrorHandler>::new(Some(1), 0.001);
    let mut traj = Trajectory::new(Some(1));
    otg.calculate(input, &mut traj)
        .unwrap_or_else(|e| panic!("calculation failed ({e:?}) for {input}"));
    traj
}

/// Position and velocity of DoF 0 at `t`.
fn state_at(traj: &Trajectory<0>, t: f64) -> (f64, f64) {
    let mut p = DataArrayOrVec::<f64, 0>::new(Some(1), 0.0);
    let mut v = DataArrayOrVec::<f64, 0>::new(Some(1), 0.0);
    traj.at_time(
        t,
        &mut Some(&mut p),
        &mut Some(&mut v),
        &mut None,
        &mut None,
        &mut None,
    );
    (p[0], v[0])
}

/// The lowest and highest position sampled every `dt` (and at the end).
fn sampled(traj: &Trajectory<0>, dt: f64) -> (f64, f64) {
    let duration = traj.get_duration();
    let n = (duration / dt).ceil() as usize;
    let mut p = DataArrayOrVec::<f64, 0>::new(Some(1), 0.0);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..=n {
        let t = (i as f64 * dt).min(duration);
        traj.at_time(
            t,
            &mut Some(&mut p),
            &mut None,
            &mut None,
            &mut None,
            &mut None,
        );
        lo = lo.min(p[0]);
        hi = hi.max(p[0]);
    }
    (lo, hi)
}

/// Assert that the extrema bound every sample, are reached by the samples, and
/// (for single-section trajectories) that the trajectory is there at their times.
fn assert_extrema_match_samples(
    traj: &mut Trajectory<0>,
    dt: f64,
    check_times: bool,
    context: &str,
) {
    let ext = traj.get_position_extrema()[0];
    let (lo, hi) = sampled(traj, dt);
    assert!(
        ext.max >= hi - 1e-9,
        "{context}: max extremum {} below sampled max {hi}",
        ext.max
    );
    assert!(
        ext.min <= lo + 1e-9,
        "{context}: min extremum {} above sampled min {lo}",
        ext.min
    );
    assert!(
        ext.max - hi < SAMPLE_TOL,
        "{context}: max extremum {} beyond sampled max {hi}",
        ext.max
    );
    assert!(
        lo - ext.min < SAMPLE_TOL,
        "{context}: min extremum {} beyond sampled min {lo}",
        ext.min
    );
    if check_times {
        let (p_max, _) = state_at(traj, ext.t_max);
        let (p_min, _) = state_at(traj, ext.t_min);
        assert!(
            (p_max - ext.max).abs() < 1e-9,
            "{context}: position {p_max} at t_max {} is not the max {}",
            ext.t_max,
            ext.max
        );
        assert!(
            (p_min - ext.min).abs() < 1e-9,
            "{context}: position {p_min} at t_min {} is not the min {}",
            ext.t_min,
            ext.min
        );
    }
}

#[test]
fn test_reversal_at_acceleration_limit_max() {
    // Moving up at 50, sent to -10: braking at a constant -100 (the jerk is high,
    // so almost all of it is a zero-jerk phase) turns at about
    // 0.05 + 49.95^2 / 200 = 12.525.
    let mut traj = plan(&input_1dof(0.0, 50.0, 0.0, -10.0, 100.0, 100.0, 100_000.0));
    let ext = traj.get_position_extrema()[0];
    let (lo, hi) = sampled(&traj, 1e-6);
    assert!((ext.max - 12.525).abs() < 1e-3, "max {}", ext.max);
    assert!((ext.max - hi).abs() < 1e-6, "max {} sampled {hi}", ext.max);
    assert!((ext.min - lo).abs() < 1e-9, "min {} sampled {lo}", ext.min);
    assert!((ext.min - -10.0).abs() < 1e-9, "min {}", ext.min);

    // At its time the trajectory is there, standing still.
    let (p, v) = state_at(&traj, ext.t_max);
    assert!((p - ext.max).abs() < 1e-9, "position {p} at t_max");
    assert!(v.abs() < 1e-9, "velocity {v} at t_max");
}

#[test]
fn test_reversal_at_acceleration_limit_min() {
    // The mirror image: moving down at 50, sent to +10.
    let mut traj = plan(&input_1dof(0.0, -50.0, 0.0, 10.0, 100.0, 100.0, 100_000.0));
    let ext = traj.get_position_extrema()[0];
    let (lo, hi) = sampled(&traj, 1e-6);
    assert!((ext.min - -12.525).abs() < 1e-3, "min {}", ext.min);
    assert!((ext.min - lo).abs() < 1e-6, "min {} sampled {lo}", ext.min);
    assert!((ext.max - hi).abs() < 1e-9, "max {} sampled {hi}", ext.max);
    assert!((ext.max - 10.0).abs() < 1e-9, "max {}", ext.max);

    let (p, v) = state_at(&traj, ext.t_min);
    assert!((p - ext.min).abs() < 1e-9, "position {p} at t_min");
    assert!(v.abs() < 1e-9, "velocity {v} at t_min");
}

#[test]
fn test_reversal_second_order() {
    // Without a jerk limit every phase has zero jerk: 50^2 / 200 = 12.5 exactly.
    let mut traj = plan(&input_1dof(
        0.0,
        50.0,
        0.0,
        -10.0,
        100.0,
        100.0,
        f64::INFINITY,
    ));
    let ext = traj.get_position_extrema()[0];
    assert!((ext.max - 12.5).abs() < 1e-9, "max {}", ext.max);
    assert!((ext.t_max - 0.5).abs() < 1e-9, "t_max {}", ext.t_max);
    assert_extrema_match_samples(&mut traj, 1e-5, true, "second order");
}

#[test]
fn test_extrema_random_states() {
    let mut rng = Pcg64Mcg::new(0x5eed);
    let pos = Uniform::new(-10.0, 10.0);
    let vel = Uniform::new(-8.0, 8.0);
    let lim = Uniform::new(0.5, 20.0);
    let jerk = Uniform::new(5.0, 1000.0);
    let unit = Uniform::new(0.0, 1.0);

    for case in 0..2000 {
        let v_max = lim.sample(&mut rng) + 5.0;
        let a_max = lim.sample(&mut rng) + 1.0;
        // Start acceleration inside the limit; zero for a quarter of the cases
        let a0 = if unit.sample(&mut rng) < 0.25 {
            0.0
        } else {
            a_max * (2.0 * unit.sample(&mut rng) - 1.0)
        };
        // A fifth of the cases are second order (zero jerk throughout)
        let j_max = if unit.sample(&mut rng) < 0.2 {
            f64::INFINITY
        } else {
            jerk.sample(&mut rng)
        };
        let input = input_1dof(
            pos.sample(&mut rng),
            vel.sample(&mut rng),
            a0,
            pos.sample(&mut rng),
            v_max,
            a_max,
            j_max,
        );
        let mut traj = plan(&input);
        assert_extrema_match_samples(&mut traj, 1e-4, true, &format!("case {case} ({input})"));
    }
}

#[test]
fn test_extrema_with_brake_trajectory() {
    let mut rng = Pcg64Mcg::new(0xb4a7e);
    let pos = Uniform::new(-10.0, 10.0);
    let lim = Uniform::new(1.0, 10.0);
    let over = Uniform::new(1.05, 3.0);
    let jerk = Uniform::new(5.0, 500.0);
    let unit = Uniform::new(0.0, 1.0);

    let mut with_brake = 0;
    for case in 0..400 {
        let v_max = lim.sample(&mut rng);
        let a_max = lim.sample(&mut rng);
        let sign = |r: f64| if r < 0.5 { -1.0 } else { 1.0 };
        // Velocity above the limit, acceleration above the limit, or both
        let (v0, a0) = match case % 3 {
            0 => (
                sign(unit.sample(&mut rng)) * v_max * over.sample(&mut rng),
                0.0,
            ),
            1 => (
                v_max * (2.0 * unit.sample(&mut rng) - 1.0),
                sign(unit.sample(&mut rng)) * a_max * over.sample(&mut rng),
            ),
            _ => (
                sign(unit.sample(&mut rng)) * v_max * over.sample(&mut rng),
                sign(unit.sample(&mut rng)) * a_max * over.sample(&mut rng),
            ),
        };
        let input = input_1dof(
            pos.sample(&mut rng),
            v0,
            a0,
            pos.sample(&mut rng),
            v_max,
            a_max,
            jerk.sample(&mut rng),
        );
        let mut traj = plan(&input);
        if traj.get_profiles()[0][0].brake.duration > 0.0 {
            with_brake += 1;
        }
        assert_extrema_match_samples(&mut traj, 1e-4, true, &format!("case {case} ({input})"));
    }
    assert!(
        with_brake > 300,
        "only {with_brake} cases had a brake pre-trajectory"
    );
}

#[test]
fn test_extrema_with_waypoints() {
    let mut rng = Pcg64Mcg::new(0x3a7);
    let pos = Uniform::new(-5.0, 5.0);
    let vel = Uniform::new(-4.0, 4.0);
    let lim = Uniform::new(1.0, 10.0);
    let jerk = Uniform::new(10.0, 1000.0);
    let count = Uniform::new(1_usize, 4);

    for case in 0..200 {
        let mut input = input_1dof(
            pos.sample(&mut rng),
            vel.sample(&mut rng),
            0.0,
            pos.sample(&mut rng),
            lim.sample(&mut rng) + 2.0,
            lim.sample(&mut rng),
            jerk.sample(&mut rng),
        );
        for _ in 0..count.sample(&mut rng) {
            input
                .intermediate_positions
                .push(DataArrayOrVec::new(Some(1), pos.sample(&mut rng)));
        }
        let mut otg = Ruckig::<0, ThrowErrorHandler>::new(Some(1), 0.001);
        let mut traj = Trajectory::new(Some(1));
        if otg.calculate(&input, &mut traj).is_err() {
            continue;
        }
        // The times of the extrema of later sections are section-local, so only
        // the values are checked here.
        assert_extrema_match_samples(&mut traj, 1e-4, false, &format!("case {case} ({input})"));
    }
}
