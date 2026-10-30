use rsruckig::prelude::*;

use float_eq::assert_float_eq;
use rsruckig::input_parameter::{ControlInterface, DurationDiscretization, Synchronization};
use rsruckig::trajectory::Trajectory;

fn almost_equal_vecs(a: &[f64], b: &[f64], epsilon: f64) -> bool {
    if a.len() != b.len() {
        panic!(
            "Length mismatch: left vector has length {}, right vector has length {}",
            a.len(),
            b.len()
        );
    }

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if (x - y).abs() > epsilon {
            panic!(
                "Values at index {} differ: left = {}, right = {}, difference = {}",
                i,
                x,
                y,
                (x - y).abs()
            );
        }
    }

    true
}

#[test]
// Single DOF
fn test_at_time() {
    // Setup
    let mut otg = Ruckig::<1, ThrowErrorHandler>::new(None, 0.005);
    let mut input = InputParameter::new(None);
    input.current_position = DataArrayOrVec::Stack([0.0]);
    input.target_position = DataArrayOrVec::Stack([1.0]);
    input.max_velocity = DataArrayOrVec::Stack([1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0]);
    input.max_jerk = DataArrayOrVec::Stack([1.0]);

    let mut traj = Trajectory::new(None);
    let _result = otg.calculate(&input, &mut traj);

    let mut output = OutputParameter::new(None);

    // Call the method you want to test
    let result = otg.update(&input, &mut output);

    // Assertions
    assert!(result.is_ok());
    assert_float_eq!(traj.get_duration(), 3.1748, abs <= 0.000_1);

    let mut new_position = DataArrayOrVec::Stack([0.0; 1]);
    let mut new_velocity = DataArrayOrVec::Stack([0.0; 1]);
    let mut new_acceleration = DataArrayOrVec::Stack([0.0; 1]);
    traj.at_time(
        0.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );
    assert_float_eq!(new_position[0], input.current_position[0], abs <= 0.000_1);

    traj.at_time(
        3.1748 / 2.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );
    assert_float_eq!(new_position[0], 0.5, abs <= 0.000_1);
}

// Secondary features
#[test]
fn test_secondary() {
    // Setup
    let mut otg = Ruckig::<3, ThrowErrorHandler>::new(None, 0.005);
    let mut input = InputParameter::new(None);
    let mut output = OutputParameter::new(None);

    input.current_position = DataArrayOrVec::Stack([0.0, -2.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([1.0, -3.0, 2.0]);
    input.target_velocity = DataArrayOrVec::Stack([0.0, 0.3, 0.0]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.max_velocity = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);

    let mut traj = Trajectory::new(None);

    let result = otg.calculate(&input, &mut traj);
    assert!(result.is_ok());
    assert_float_eq!(traj.get_duration(), 4.0, abs <= 0.000_1);

    let result = otg.update(&input, &mut output);

    assert!(result.is_ok());
    assert_float_eq!(output.trajectory.get_duration(), 4.0, abs <= 0.000_1);

    let mut new_position = DataArrayOrVec::Stack([0.0; 3]);
    let mut new_velocity = DataArrayOrVec::Stack([0.0; 3]);
    let mut new_acceleration = DataArrayOrVec::Stack([0.0; 3]);
    let mut new_jerk = DataArrayOrVec::Stack([0.0; 3]);

    output.trajectory.at_time(
        0.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );
    assert_eq!(new_position, input.current_position);
    assert_eq!(new_velocity, input.current_velocity);
    assert_eq!(new_acceleration, input.current_acceleration);

    let mut new_section: Option<usize> = None;

    output.trajectory.at_time(
        output.trajectory.get_duration(),
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );
    assert!(almost_equal_vecs(
        &new_position,
        &input.target_position,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &new_velocity,
        &input.target_velocity,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &new_acceleration,
        &input.target_acceleration,
        0.000_1,
    ));

    output.trajectory.at_time(
        2.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut Some(&mut new_jerk),
        &mut new_section,
    );

    assert!(almost_equal_vecs(
        &new_position,
        &[0.5, -2.6871268303003437, 1.0],
        0.000_1,
    ));
    assert_eq!(new_jerk, DataArrayOrVec::Stack([0.0, 0.0, -1.0]));
    assert_eq!(new_section, Some(0));

    output.trajectory.at_time(
        5.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut Some(&mut new_jerk),
        &mut new_section,
    );
    assert_eq!(new_jerk, DataArrayOrVec::Stack([0.0, 0.0, 0.0]));
    assert_eq!(new_section, Some(1));

    let independent_min_durations = output.trajectory.get_independent_min_durations();
    assert_float_eq!(independent_min_durations[0], 3.1748021039, abs <= 0.000_1);
    assert_float_eq!(independent_min_durations[1], 3.6860977315, abs <= 0.000_1);
    assert_float_eq!(
        independent_min_durations[2],
        output.trajectory.get_duration(),
        abs <= 0.000_1
    );

    let position_extrema = output.trajectory.get_position_extrema();
    assert_float_eq!(position_extrema[0].t_max, 4.0, abs <= 0.000_1);
    assert_float_eq!(position_extrema[0].max, 1.0, abs <= 0.000_1);
    assert_float_eq!(position_extrema[0].t_min, 0.0, abs <= 0.000_1);
    assert_float_eq!(position_extrema[0].min, 0.0, abs <= 0.000_1);

    assert_float_eq!(position_extrema[1].t_max, 0.0, abs <= 0.000_1);
    assert_float_eq!(position_extrema[1].max, -2.0, abs <= 0.000_1);
    assert_float_eq!(position_extrema[1].t_min, 3.2254033308, abs <= 0.000_1);
    assert_float_eq!(position_extrema[1].min, -3.1549193338, abs <= 0.000_1);

    assert_float_eq!(position_extrema[2].t_max, 4.0, abs <= 0.000_1);
    assert_float_eq!(position_extrema[2].max, 2.0, abs <= 0.000_1);
    assert_float_eq!(position_extrema[2].t_min, 0.0, abs <= 0.000_1);
    assert_float_eq!(position_extrema[2].min, 0.0, abs <= 0.000_1);

    let time = output.trajectory.get_first_time_at_position(0, 0.0);
    assert_ne!(time, None);
    assert_float_eq!(time.unwrap(), 0.0, abs <= 0.000_1);

    let time = output.trajectory.get_first_time_at_position(0, 0.5);
    assert_ne!(time, None);
    assert_float_eq!(time.unwrap(), 2.0, abs <= 0.000_1);

    let time = output.trajectory.get_first_time_at_position(0, 1.0);
    assert_ne!(time, None);
    assert_float_eq!(time.unwrap(), 4.0, abs <= 0.000_1);

    let time = output.trajectory.get_first_time_at_position(1, -3.0);
    assert_ne!(time, None);
    assert_float_eq!(time.unwrap(), 2.6004877902, abs <= 0.000_1);

    let time = output.trajectory.get_first_time_at_position(1, -3.1);
    assert_ne!(time, None);
    assert_float_eq!(time.unwrap(), 2.8644154489, abs <= 0.000_1);

    let time = output.trajectory.get_first_time_at_position(2, 0.05);
    assert_ne!(time, None);
    assert_float_eq!(time.unwrap(), 0.6694329501, abs <= 0.000_1);

    let time = output.trajectory.get_first_time_at_position(0, -1.0);
    assert_eq!(time, None);

    let time = output.trajectory.get_first_time_at_position(1, -3.4);
    assert_eq!(time, None);

    let time = output.trajectory.get_first_time_at_position(6, 0.0);
    assert_eq!(time, None);
    input.current_position = DataArrayOrVec::Stack([0.0, -2.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([1.0, -3.0, 2.0]);
    input.target_velocity = DataArrayOrVec::Stack([2.0, 0.3, 0.0]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.max_velocity = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);

    let result = otg.update(&input, &mut output);

    match result {
        Ok(_) => panic!("Expected an error but got a successful result."),
        Err(e) => {
            let error_message = e.to_string();
            assert!(
                error_message.contains("exceeds its maximum velocity limit"),
                "Unexpected error message: {}",
                error_message
            );
        }
    }
    assert!(!output.new_calculation);

    // Modify input parameters to trigger a new calculation
    input.target_velocity = DataArrayOrVec::Stack([0.2, -0.3, 0.8]);
    let result = otg.update(&input, &mut output);
    assert!(result.is_ok());
    assert!(output.new_calculation);

    input.minimum_duration = Some(12.0);
    let result = otg.update(&input, &mut output);
    assert!(result.is_ok());
    assert_float_eq!(output.trajectory.get_duration(), 12.0, abs <= 0.000_1);

    input.current_position = DataArrayOrVec::Stack([1300.0, 0.0, 0.02]);
    input.current_velocity = DataArrayOrVec::Stack([1200.0, 0.0, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([1400.0, 0.0, 0.02]);
    input.target_velocity = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.max_velocity = DataArrayOrVec::Stack([800.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([40000.0, 1.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Stack([200000.0, 1.0, 1.0]);

    input.minimum_duration = None;

    let result = otg.update(&input, &mut output);

    assert!(result.is_ok());
    assert_float_eq!(output.trajectory.get_duration(), 0.167347, abs <= 0.000_1);

    let independent_min_durations = output.trajectory.get_independent_min_durations();
    assert_float_eq!(
        independent_min_durations[0],
        output.trajectory.get_duration(),
        abs <= 0.000_1
    );
    assert_float_eq!(independent_min_durations[1], 0.0, abs <= 0.000_1);
    assert_float_eq!(independent_min_durations[2], 0.0, abs <= 0.000_1);
}

#[test]
fn test_enabled() {
    // Setup
    let mut otg = Ruckig::<3, ThrowErrorHandler>::new(None, 0.005);
    let mut input = InputParameter::new(None);
    let mut output = OutputParameter::new(None);

    input.enabled = DataArrayOrVec::Stack([true, false, false]);
    input.current_position = DataArrayOrVec::Stack([0.0, -2.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([0.0, 0.1, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([0.0, 0.0, -0.2]);

    input.target_position = DataArrayOrVec::Stack([1.0, -3.0, 2.0]);

    input.max_velocity = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);

    let result = otg.update(&input, &mut output);

    assert!(result.is_ok());
    assert_float_eq!(
        output.trajectory.get_duration(),
        3.1748021039,
        abs <= 0.000_1
    );

    let mut new_position = DataArrayOrVec::Stack([0.0; 3]);
    let mut new_velocity = DataArrayOrVec::Stack([0.0; 3]);
    let mut new_acceleration = DataArrayOrVec::Stack([0.0; 3]);

    output.trajectory.at_time(
        0.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );

    assert!(almost_equal_vecs(
        &new_position,
        &input.current_position,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &new_velocity,
        &input.current_velocity,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &new_acceleration,
        &input.current_acceleration,
        0.000_1,
    ));

    output.trajectory.at_time(
        output.trajectory.get_duration(),
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );
    assert!(almost_equal_vecs(
        &new_position,
        &[input.target_position[0], -1.6825197896, -1.0079368399],
        0.000_1,
    ));

    // Make sure that disabled DoFs overwrite prior blocks
    input.enabled = DataArrayOrVec::Stack([true, true, true]);
    input.current_position = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.target_position = DataArrayOrVec::Stack([100.0, -3000.0, 2000.0]);
    input.target_velocity = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);

    let _result = otg.update(&input, &mut output);

    input.enabled = DataArrayOrVec::Stack([false, false, true]);
    input.current_position = DataArrayOrVec::Stack([0.0, -2.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([0.0, 0.2, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([0.0, 0.2, 0.0]);

    input.target_position = DataArrayOrVec::Stack([1.0, -3.0, 2.0]);
    input.target_velocity = DataArrayOrVec::Stack([0.0, 0.0, 0.2]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, -0.1]);

    input.max_velocity = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);

    let result = otg.update(&input, &mut output);

    assert!(result.is_ok());
    assert_float_eq!(
        output.trajectory.get_duration(),
        3.6578610221,
        abs <= 0.000_1
    );
}

#[test]
fn test_phase_synchronization() {
    // Setup
    let mut otg = Ruckig::<3, ThrowErrorHandler>::new(None, 0.005);
    let mut input = InputParameter::new(None);
    let mut output = OutputParameter::new(None);
    let mut traj = Trajectory::new(None);

    let mut new_position = DataArrayOrVec::Stack([0.0; 3]);
    let mut new_velocity = DataArrayOrVec::Stack([0.0; 3]);
    let mut new_acceleration = DataArrayOrVec::Stack([0.0; 3]);

    input.current_position = DataArrayOrVec::Stack([0.0, -2.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([1.0, -3.0, 2.0]);

    input.max_velocity = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);

    input.synchronization = Synchronization::Phase;

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert!(almost_equal_vecs(
        &traj.get_profiles()[0][0].t,
        &traj.get_profiles()[0][1].t,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &traj.get_profiles()[0][0].t,
        &traj.get_profiles()[0][2].t,
        0.000_1,
    ));

    let result = otg.update(&input, &mut output);

    output.trajectory.at_time(
        1.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );
    assert!(result.is_ok());
    assert_float_eq!(output.trajectory.get_duration(), 4.0, abs <= 0.000_1);
    assert!(almost_equal_vecs(
        &new_position,
        &[0.0833333333, -2.0833333333, 0.1666666667],
        0.000_1,
    ));

    assert!(almost_equal_vecs(
        &output.trajectory.get_profiles()[0][0].t,
        &output.trajectory.get_profiles()[0][1].t,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &output.trajectory.get_profiles()[0][0].t,
        &output.trajectory.get_profiles()[0][2].t,
        0.000_1,
    ));

    input.current_position = DataArrayOrVec::Stack([0.0, -2.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([10.0, -3.0, 2.0]);

    input.max_velocity = DataArrayOrVec::Stack([10.0, 2.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([10.0, 2.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Stack([10.0, 2.0, 1.0]);

    let result = otg.update(&input, &mut output);

    output.trajectory.at_time(
        1.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );

    assert!(result.is_ok());
    assert_float_eq!(output.trajectory.get_duration(), 4.0, abs <= 0.000_1);
    assert!(almost_equal_vecs(
        &new_position,
        &[0.8333333333, -2.0833333333, 0.1666666667],
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &output.trajectory.get_profiles()[0][0].t,
        &output.trajectory.get_profiles()[0][1].t,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &output.trajectory.get_profiles()[0][0].t,
        &output.trajectory.get_profiles()[0][2].t,
        0.000_1,
    ));

    // Test equal start and target state
    input.current_position = DataArrayOrVec::Stack([1.0, -2.0, 3.0]);
    input.target_position = DataArrayOrVec::Stack([1.0, -2.0, 3.0]);

    let result = otg.update(&input, &mut output);
    output.trajectory.at_time(
        0.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );

    assert!(result.is_ok());
    assert_float_eq!(output.trajectory.get_duration(), 0.0, abs <= 0.000_1);
    assert!(almost_equal_vecs(&new_position, &[1.0, -2.0, 3.0], 0.000_1));
    assert!(almost_equal_vecs(
        &output.trajectory.get_profiles()[0][0].t,
        &output.trajectory.get_profiles()[0][1].t,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &output.trajectory.get_profiles()[0][0].t,
        &output.trajectory.get_profiles()[0][2].t,
        0.000_1,
    ));

    input.current_position = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.target_velocity = DataArrayOrVec::Stack([0.2, 0.3, 0.4]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.max_velocity = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);

    let result = otg.calculate(&input, &mut traj);
    assert!(result.is_ok());
    assert!(almost_equal_vecs(
        &traj.get_profiles()[0][0].t,
        &traj.get_profiles()[0][1].t,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &traj.get_profiles()[0][0].t,
        &traj.get_profiles()[0][2].t,
        0.000_1,
    ));

    input.current_position = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([0.0, 0.0, 0.01]);
    input.target_velocity = DataArrayOrVec::Stack([0.2, 0.3, 0.4]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert_ne!(traj.get_profiles()[0][0].t, traj.get_profiles()[0][1].t);
    assert_ne!(traj.get_profiles()[0][0].t, traj.get_profiles()[0][2].t);

    input.current_position = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([0.4, 0.15, 0.2]);
    input.current_acceleration = DataArrayOrVec::Stack([0.8, 0.3, 0.4]);

    input.target_position = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.target_velocity = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert!(almost_equal_vecs(
        &traj.get_profiles()[0][0].t,
        &traj.get_profiles()[0][1].t,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &traj.get_profiles()[0][0].t,
        &traj.get_profiles()[0][2].t,
        0.000_1,
    ));

    input.max_velocity = DataArrayOrVec::Stack([1.0, 0.2, 1.0]);

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert_ne!(traj.get_profiles()[0][0].t, traj.get_profiles()[0][1].t);
    assert_ne!(traj.get_profiles()[0][0].t, traj.get_profiles()[0][2].t);

    input.current_position = DataArrayOrVec::Stack([0.0, 0.02, 1.0]);
    input.current_velocity = DataArrayOrVec::Stack([-0.2, 0.15, 0.2]);
    input.current_acceleration = DataArrayOrVec::Stack([-0.4, 0.3, 0.4]);

    input.target_position = DataArrayOrVec::Stack([0.03, 0.0, 0.0]);
    input.target_velocity = DataArrayOrVec::Stack([-0.02, 0.015, 0.02]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.max_velocity = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.control_interface = ControlInterface::Velocity;

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert!(almost_equal_vecs(
        &traj.get_profiles()[0][0].t,
        &traj.get_profiles()[0][1].t,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &traj.get_profiles()[0][0].t,
        &traj.get_profiles()[0][2].t,
        0.000_1,
    ));

    input.max_jerk = DataArrayOrVec::Stack([1.0, 0.1, 1.0]);

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert!(almost_equal_vecs(
        &traj.get_profiles()[0][0].t,
        &traj.get_profiles()[0][1].t,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &traj.get_profiles()[0][0].t,
        &traj.get_profiles()[0][2].t,
        0.000_1,
    ));

    input.target_acceleration = DataArrayOrVec::Stack([0.01, 0.0, 0.0]);

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert_ne!(traj.get_profiles()[0][0].t, traj.get_profiles()[0][1].t);
    assert_ne!(traj.get_profiles()[0][0].t, traj.get_profiles()[0][2].t);

    input.current_position = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.target_velocity = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
}

#[test]
fn test_discretion() {
    // Setup
    let mut otg = Ruckig::<3, ThrowErrorHandler>::new(None, 0.01);
    let mut input = InputParameter::new(None);
    let mut output = OutputParameter::new(None);
    let mut traj = Trajectory::new(None);

    let mut new_position = DataArrayOrVec::Stack([0.0; 3]);
    let mut new_velocity = DataArrayOrVec::Stack([0.0; 3]);
    let mut new_acceleration = DataArrayOrVec::Stack([0.0; 3]);

    input.current_position = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([1.0, -3.0, 2.0]);
    input.target_velocity = DataArrayOrVec::Stack([0.2, 0.2, 0.2]);

    input.max_velocity = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([2.0, 2.0, 2.0]);
    input.max_jerk = DataArrayOrVec::Stack([1.8, 2.4, 2.0]);

    input.duration_discretization = DurationDiscretization::Discrete;

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert_float_eq!(traj.get_duration(), 4.5, abs <= 0.000_1);

    let _result = otg.update(&input, &mut output);
    output.trajectory.at_time(
        4.5,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );
    assert!(almost_equal_vecs(&new_position, &[1.0, -3.0, 2.0], 0.000_1));
}

#[test]
fn test_per_dof_setting() {
    let mut otg = Ruckig::<3, ThrowErrorHandler>::new(None, 0.005);
    let mut input = InputParameter::new(None);
    let mut traj = Trajectory::new(None);

    // Test case 1
    input.current_position = DataArrayOrVec::Stack([0.0, -2.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([1.0, -3.0, 2.0]);
    input.target_velocity = DataArrayOrVec::Stack([0.0, 0.3, 0.0]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.max_velocity = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);

    let result = otg.calculate(&input, &mut traj);
    assert!(result.is_ok());
    assert_float_eq!(traj.get_duration(), 4.0, abs <= 0.000_1);

    let mut new_position = DataArrayOrVec::Stack([0.0; 3]);
    let mut new_velocity = DataArrayOrVec::Stack([0.0; 3]);
    let mut new_acceleration = DataArrayOrVec::Stack([0.0; 3]);

    traj.at_time(
        2.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );
    assert!(almost_equal_vecs(
        &new_position,
        &[0.5, -2.6871268303, 1.0],
        0.000_1,
    ));

    input.control_interface = ControlInterface::Velocity;

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert_float_eq!(traj.get_duration(), 1.095445115, abs <= 0.000_1);

    traj.at_time(
        1.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );
    assert!(almost_equal_vecs(
        &new_position,
        &[0.0, -1.8641718534, 0.0],
        0.000_1,
    ));

    input.per_dof_control_interface = Some(DataArrayOrVec::Stack([
        ControlInterface::Position,
        ControlInterface::Velocity,
        ControlInterface::Position,
    ]));

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert_float_eq!(traj.get_duration(), 4.0, abs <= 0.000_1);

    input.per_dof_synchronization = Some(DataArrayOrVec::Stack([
        Synchronization::Time,
        Synchronization::None,
        Synchronization::Time,
    ]));

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert_float_eq!(traj.get_duration(), 4.0, abs <= 0.000_1);

    traj.at_time(
        2.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );

    assert!(almost_equal_vecs(
        &new_position,
        &[0.5, -1.5643167673, 1.0],
        0.000_1,
    ));

    input.control_interface = ControlInterface::Position;
    input.per_dof_control_interface = None;
    input.per_dof_synchronization = Some(DataArrayOrVec::Stack([
        Synchronization::None,
        Synchronization::Time,
        Synchronization::Time,
    ]));

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert_float_eq!(traj.get_duration(), 4.0, abs <= 0.000_1);

    traj.at_time(
        2.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );

    assert!(almost_equal_vecs(
        &new_position,
        &[0.7482143874, -2.6871268303, 1.0],
        0.000_1,
    ));

    let independent_min_durations = traj.get_independent_min_durations();

    traj.at_time(
        independent_min_durations[0],
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );

    assert_float_eq!(new_position[0], input.target_position[0], abs <= 0.000_1);

    traj.at_time(
        independent_min_durations[1],
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );

    assert_float_eq!(new_position[1], -3.0890156397, abs <= 0.000_1);

    traj.at_time(
        independent_min_durations[2],
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );

    assert_float_eq!(new_position[2], input.target_position[2], abs <= 0.000_1);

    input.current_position = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([35.0, 35.0, 35.0]);
    input.target_velocity = DataArrayOrVec::Stack([125.0, 125.0, 100.0]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.max_velocity = DataArrayOrVec::Stack([125.0, 125.0, 100.0]);
    input.max_acceleration = DataArrayOrVec::Stack([2000.0, 2000.0, 2000.0]);
    input.max_jerk = DataArrayOrVec::Stack([20000.0, 20000.0, 20000.0]);

    input.per_dof_synchronization = Some(DataArrayOrVec::Stack([
        Synchronization::Time,
        Synchronization::Time,
        Synchronization::None,
    ]));

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert_float_eq!(traj.get_duration(), 0.4207106781, abs <= 0.000_1);

    input.current_position = DataArrayOrVec::Stack([0.0, -2.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([0.0, 0.2, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([0.0, 0.2, 0.0]);

    input.target_position = DataArrayOrVec::Stack([1.0, -3.0, 2.0]);
    input.target_velocity = DataArrayOrVec::Stack([0.0, 0.0, 0.2]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, -0.1]);

    input.max_velocity = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);

    input.per_dof_synchronization = Some(DataArrayOrVec::Stack([
        Synchronization::None,
        Synchronization::None,
        Synchronization::Time,
    ]));

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert_float_eq!(traj.get_duration(), 3.7885667284, abs <= 0.000_1);

    input.per_dof_synchronization = Some(DataArrayOrVec::Stack([
        Synchronization::None,
        Synchronization::Time,
        Synchronization::None,
    ]));

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert_float_eq!(traj.get_duration(), 3.7885667284, abs <= 0.000_1);

    input.enabled = DataArrayOrVec::Stack([true, false, true]);

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert_float_eq!(traj.get_duration(), 3.6578610221, abs <= 0.000_1);

    input.current_position = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([0.2, 0.0, -0.1]);
    input.current_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([1.0, -0.2, -0.5]);
    input.target_velocity = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.max_velocity = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);

    input.per_dof_synchronization = Some(DataArrayOrVec::Stack([
        Synchronization::Phase,
        Synchronization::None,
        Synchronization::Phase,
    ]));

    input.enabled = DataArrayOrVec::Stack([true, true, true]);

    let result = otg.calculate(&input, &mut traj);

    assert!(result.is_ok());
    assert_float_eq!(traj.get_duration(), 2.848387279, abs <= 0.000_1);
    assert_ne!(traj.get_profiles()[0][0].t, traj.get_profiles()[0][1].t);
    assert!(almost_equal_vecs(
        &traj.get_profiles()[0][0].t,
        &traj.get_profiles()[0][2].t,
        0.000_1,
    ));
}

#[test]
fn test_dynamic_dofs() {
    let mut otg = Ruckig::<0, ThrowErrorHandler>::new(Some(3), 0.005);
    let mut input = InputParameter::new(Some(3));
    let mut output = OutputParameter::new(Some(3));

    input.current_position = DataArrayOrVec::Heap(vec![0.0, -2.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Heap(vec![0.0, 0.0, 0.0]);
    input.current_acceleration = DataArrayOrVec::Heap(vec![0.0, 0.0, 0.0]);

    input.target_position = DataArrayOrVec::Heap(vec![1.0, -3.0, 2.0]);
    input.target_velocity = DataArrayOrVec::Heap(vec![0.0, 0.3, 0.0]);
    input.target_acceleration = DataArrayOrVec::Heap(vec![0.0, 0.0, 0.0]);

    input.max_velocity = DataArrayOrVec::Heap(vec![1.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Heap(vec![1.0, 1.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Heap(vec![1.0, 1.0, 1.0]);

    let result = otg.update(&input, &mut output);

    assert!(result.is_ok());
    assert_float_eq!(output.trajectory.get_duration(), 4.0, abs <= 0.000_1);

    let mut new_position = DataArrayOrVec::Heap(vec![0.0; 3]);
    let mut new_velocity = DataArrayOrVec::Heap(vec![0.0; 3]);
    let mut new_acceleration = DataArrayOrVec::Heap(vec![0.0; 3]);

    output.trajectory.at_time(
        0.0,
        &mut Some(&mut new_position),
        &mut Some(&mut new_velocity),
        &mut Some(&mut new_acceleration),
        &mut None,
        &mut None,
    );
    assert!(almost_equal_vecs(
        &new_position,
        &input.current_position,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &new_velocity,
        &input.current_velocity,
        0.000_1,
    ));
    assert!(almost_equal_vecs(
        &new_acceleration,
        &input.current_acceleration,
        0.000_1,
    ));
}

#[test]
fn test_zero_limits() {
    let mut otg = Ruckig::<3, ThrowErrorHandler>::new(None, 0.005);
    let mut input = InputParameter::new(None);
    let mut output = OutputParameter::new(None);

    input.current_position = DataArrayOrVec::Stack([0.0, -2.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([0.2, 0.0, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([1.0, -3.0, 0.0]);
    input.target_velocity = DataArrayOrVec::Stack([0.2, 0.0, 0.0]);
    input.target_acceleration = DataArrayOrVec::Stack([0.0, 0.0, 0.0]);

    input.max_velocity = DataArrayOrVec::Stack([1.0, 1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([0.0, 1.0, 0.0]);
    input.max_jerk = DataArrayOrVec::Stack([0.0, 1.0, 0.0]);

    let result = otg.update(&input, &mut output);

    assert!(result.is_ok());
    assert_float_eq!(output.trajectory.get_duration(), 5.0, abs <= 0.000_1);

    input.current_position = DataArrayOrVec::Stack([0.0, -2.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([-0.2, 0.0, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([1.0, 0.0, 0.0]);

    input.target_position = DataArrayOrVec::Stack([0.4, -3.0, 0.0]);
    input.target_velocity = DataArrayOrVec::Stack([0.8, 0.0, 0.0]);
    input.target_acceleration = DataArrayOrVec::Stack([1.0, 0.0, 0.0]);

    input.max_velocity = DataArrayOrVec::Stack([1.0, 200.0, 0.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 200.0, 0.0]);
    input.max_jerk = DataArrayOrVec::Stack([0.0, 200.0, 0.0]);

    let result = otg.update(&input, &mut output);

    match result {
        Ok(_) => panic!("Expected an error but got a successful result."),
        Err(e) => {
            let error_message = e.to_string();
            assert!(
                error_message.contains("zero limits conflict in step 1"),
                "Unexpected error message: {}",
                error_message
            );
        }
    }

    input.target_position = DataArrayOrVec::Stack([0.3, -3.0, 0.0]);
    input.max_velocity = DataArrayOrVec::Stack([1.0, 2.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 2.0, 0.0]);
    input.max_jerk = DataArrayOrVec::Stack([0.0, 2.0, 0.0]);

    let result = otg.update(&input, &mut output);

    match result {
        Ok(_) => panic!("Expected an error but got a successful result."),
        Err(e) => {
            let error_message = e.to_string();
            assert!(
                error_message.contains("zero limits conflict with other"),
                "Unexpected error message: {}",
                error_message
            );
        }
    }

    input.control_interface = ControlInterface::Velocity;
    input.current_position = DataArrayOrVec::Stack([0.0, -2.0, 0.0]);
    input.current_velocity = DataArrayOrVec::Stack([-0.2, 0.0, 0.0]);
    input.current_acceleration = DataArrayOrVec::Stack([1.0, 0.0, 0.2]);
    input.target_position = DataArrayOrVec::Stack([0.4, -3.0, 0.0]);
    input.target_velocity = DataArrayOrVec::Stack([0.9, 0.5, 0.4]);
    input.target_acceleration = DataArrayOrVec::Stack([1.0, 0.0, 0.2]);
    input.max_velocity = DataArrayOrVec::Stack([1.0, 2.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 2.0, 6.0]);
    input.max_jerk = DataArrayOrVec::Stack([0.0, 2.0, 0.0]);

    let result = otg.update(&input, &mut output);

    match result {
        Ok(_) => panic!("Expected an error but got a successful result."),
        Err(e) => {
            let error_message = e.to_string();
            assert!(
                error_message.contains("zero limits conflict with other"),
                "Unexpected error message: {}",
                error_message
            );
        }
    }

    input.max_jerk = DataArrayOrVec::Stack([1.0, 2.0, 0.0]);

    let result = otg.update(&input, &mut output);

    assert!(result.is_ok());
    assert_float_eq!(output.trajectory.get_duration(), 2.0, abs <= 0.000_1);

    input.max_jerk = DataArrayOrVec::Stack([0.0, 2.0, 20.0]);

    let result = otg.update(&input, &mut output);

    assert!(result.is_ok());
    assert_float_eq!(output.trajectory.get_duration(), 1.1, abs <= 0.000_1);
}

#[test]
fn test_min_duration() -> Result<(), RuckigError> {
    let mut otg = Ruckig::<1, ThrowErrorHandler>::new(None, 0.01);
    let mut input = InputParameter::new(None);

    input.current_position[0] = 0.0;
    input.current_velocity[0] = 0.0;
    input.current_acceleration[0] = 0.0;

    input.target_position[0] = 1.0;
    input.target_velocity[0] = 1.0;
    input.target_acceleration[0] = 1.0;

    input.max_velocity[0] = 1.0;
    input.max_acceleration[0] = 2.0;
    input.max_jerk[0] = 3.0;

    input.minimum_duration = None;
    input.synchronization = Synchronization::None;
    let mut trajectory = Trajectory::new(None);
    otg.calculate(&input, &mut trajectory)?;

    let duration = trajectory.get_duration();

    let mut trajectory_min_duration = Trajectory::new(None);
    input.minimum_duration = Some(5.0);
    otg.calculate(&input, &mut trajectory_min_duration)?;
    let new_duration = trajectory_min_duration.duration;
    dbg!(duration, new_duration);
    assert!(new_duration > duration);
    assert_float_eq!(new_duration, 5.0, abs <= 0.000_1);

    Ok(())
}

#[test]
fn test_matched_signs_phase_sync() {
    let mut otg = Ruckig::<2, ThrowErrorHandler>::new(None, 0.01);

    let mut input = InputParameter::new(None);
    input.synchronization = Synchronization::Phase;

    // DOF0 will have positive velocity, moving from 0.0 -> 1.0
    // DOF1 will have positive velocity, moving from 0.0 -> 2.0
    input.current_position = daov_stack![0.0, 0.0];
    input.target_position = daov_stack![1.0, 2.0];

    // Start and end at standstill
    input.current_velocity = daov_stack![0.0, 0.0];
    input.target_velocity = daov_stack![0.0, 0.0];

    // Limits
    input.max_velocity = daov_stack![1.0, 1000.0];
    input.max_acceleration = daov_stack![10.0, 1000.0];

    let mut trajectory = Trajectory::new(None);

    let _result = otg
        .calculate(&input, &mut trajectory)
        .expect("This trajectory is solvable.");

    let profiles = trajectory.get_profiles().first().unwrap();
    let dof0_profile = profiles.get(0).unwrap();
    let dof1_profile = profiles.get(1).unwrap();

    assert_eq!(dof0_profile.t, dof1_profile.t);
}

#[test]
fn test_mixed_signs_phase_sync() {
    let mut otg = Ruckig::<2, ThrowErrorHandler>::new(None, 0.01);

    let mut input = InputParameter::new(None);
    input.synchronization = Synchronization::Phase;

    // DOF0 will have negative velocity, moving from 1.0 -> 0.0
    // DOF1 will have positive velocity, moving from 0.0 -> 2.0
    input.current_position = daov_stack![1.0, 0.0];
    input.target_position = daov_stack![0.0, 2.0];

    // Start and end at standstill
    input.current_velocity = daov_stack![0.0, 0.0];
    input.target_velocity = daov_stack![0.0, 0.0];

    // Limits
    input.max_velocity = daov_stack![1.0, 1000.0];
    input.max_acceleration = daov_stack![10.0, 1000.0];

    let mut trajectory = Trajectory::new(None);

    let _result = otg
        .calculate(&input, &mut trajectory)
        .expect("This trajectory is solvable.");

    let profiles = trajectory.get_profiles().first().unwrap();
    let dof0_profile = profiles.get(0).unwrap();
    let dof1_profile = profiles.get(1).unwrap();

    assert_eq!(dof0_profile.t, dof1_profile.t);
}

#[test]
fn test_degenerate_boundary_state() {
    // Regression test (GitHub issue): boundary states lying exactly on the
    // velocity-at-zero-acceleration curve (v0 == a0²/(2·j_max) and
    // vf == af²/(2·j_max)) made two step-1 profile families produce the same
    // degenerate profile with the same direction. The duplicate was not
    // merged, leaving an even profile count and failing the calculation.
    let p0 = 1.1032599505208334;
    let v0 = 0.36337812500000005;
    let a0 = 0.8525;
    let pf = 2.9033375720687795;
    let vf = 0.3477303818980908;
    let af = -0.833942902;

    // The exact input and tiny perturbations around the degeneracy must all solve
    let perturbations: [(f64, f64, f64, f64, f64, f64); 7] = [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1e-12, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1e-12, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1e-12, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1e-12, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 1e-12),
        (0.0, 0.0, 0.0, 0.0, 0.0, -1e-12),
    ];

    for (i, (dp0, dv0, da0, dpf, dvf, daf)) in perturbations.iter().enumerate() {
        let mut input = InputParameter::<1>::new(None);
        input.synchronization = Synchronization::None;
        input.current_position = DataArrayOrVec::Stack([p0 + dp0]);
        input.current_velocity = DataArrayOrVec::Stack([v0 + dv0]);
        input.current_acceleration = DataArrayOrVec::Stack([a0 + da0]);
        input.target_position = DataArrayOrVec::Stack([pf + dpf]);
        input.target_velocity = DataArrayOrVec::Stack([vf + dvf]);
        input.target_acceleration = DataArrayOrVec::Stack([af + daf]);
        input.max_velocity = DataArrayOrVec::Stack([1.0]);
        input.max_acceleration = DataArrayOrVec::Stack([1.0]);
        input.max_jerk = DataArrayOrVec::Stack([1.0]);

        let mut otg = Ruckig::<1, ThrowErrorHandler>::new(None, 0.01);
        let mut traj = Trajectory::<1>::new(None);
        let result = otg.calculate(&input, &mut traj);
        assert!(result.is_ok(), "perturbation {i} failed: {result:?}");
        assert_float_eq!(traj.get_duration(), 2.313557098, abs <= 1e-6);
    }
}

#[test]
fn test_step1_acc1_profile_at_the_limit() {
    // Regression test: with large asymmetric acceleration limits, the time-
    // optimal profile reaches the deceleration limit (ReachedLimits::ACC1),
    // but its acceleration integrated over the phase lands a few ulps past
    // that limit. C++ Ruckig sets it to the limit before the check
    // (`check<UDDU, ACC1, true>`); without that, step 1 refused the profile
    // and either found none (the second move) or a 40x slower one (the
    // first). Durations are C++ Ruckig 0.19.4's for the same inputs.
    // (p0, v0, pf, v_max, a_max, a_min, j_max, duration)
    let moves: [(f64, f64, f64, f64, f64, f64, f64, f64); 2] = [
        (
            0.0,
            13.915793826757358,
            5.613113707547784,
            1455.4579631565307,
            240638.6265394368,
            -104670.71916922588,
            24541362.601308085,
            0.019311372315759107,
        ),
        (
            0.0,
            0.0,
            -28.240860972682604,
            4535.282836705683,
            106701.04533749608,
            -125440.6060879475,
            11435757.787945986,
            0.043049204455377126,
        ),
    ];

    for (i, &(p0, v0, pf, v_max, a_max, a_min, j_max, duration)) in moves.iter().enumerate() {
        let mut input = InputParameter::<1>::new(None);
        input.current_position = DataArrayOrVec::Stack([p0]);
        input.current_velocity = DataArrayOrVec::Stack([v0]);
        input.target_position = DataArrayOrVec::Stack([pf]);
        input.max_velocity = DataArrayOrVec::Stack([v_max]);
        input.min_velocity = Some(DataArrayOrVec::Stack([-v_max]));
        input.max_acceleration = DataArrayOrVec::Stack([a_max]);
        input.min_acceleration = Some(DataArrayOrVec::Stack([a_min]));
        input.max_jerk = DataArrayOrVec::Stack([j_max]);

        let mut otg = Ruckig::<1, ThrowErrorHandler>::new(None, 0.001);
        let mut traj = Trajectory::<1>::new(None);
        let result = otg.calculate(&input, &mut traj);
        assert!(result.is_ok(), "move {i} failed: {result:?}");
        assert_float_eq!(traj.get_duration(), duration, rmax <= 1e-12);
    }
}

// Regression test for issue #16: `output.time` must be reset to zero when
// `update()` recalculates the trajectory (e.g. on a mid-flight retarget).
// Otherwise the new trajectory — whose time origin is the current state — is
// sampled at the old accumulated time, causing a discontinuous position jump
// and a premature `Finished`.
#[test]
fn test_time_reset_on_retarget() {
    let dt = 0.01;
    let mut otg = Ruckig::<1, ThrowErrorHandler>::new(None, dt);
    let mut input = InputParameter::<1>::new(None);
    let mut output = OutputParameter::<1>::new(None);

    input.target_position = DataArrayOrVec::Stack([1.0]);
    input.max_velocity = DataArrayOrVec::Stack([1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([2.0]);
    input.max_jerk = DataArrayOrVec::Stack([10.0]);

    // Drive towards the first target for 50 cycles (0.5 s).
    for _ in 0..50 {
        otg.update(&input, &mut output).unwrap();
        output.pass_to_input(&mut input);
    }
    let pos_before = output.new_position[0];

    // Change the target mid-flight -> triggers recalculation inside update().
    input.target_position = DataArrayOrVec::Stack([-1.0]);
    let mut res = otg.update(&input, &mut output).unwrap();

    // The output.time must have been reset for the new trajectory.
    assert_float_eq!(output.time, dt, abs <= 1e-12);
    assert!(output.new_calculation);

    // No discontinuous jump: within one cycle the position may not move faster
    // than max_velocity * dt (= 0.01) by more than a small margin.
    let step = (output.new_position[0] - pos_before).abs();
    assert!(
        step <= 0.02,
        "position jumped {step} in a single cycle (max_velocity * dt = {})",
        1.0 * dt
    );
    output.pass_to_input(&mut input);

    // Run to completion and confirm it does not finish before the trajectory's
    // own duration.
    let mut cycles = 1;
    while res == RuckigResult::Working && cycles < 1000 {
        res = otg.update(&input, &mut output).unwrap();
        output.pass_to_input(&mut input);
        cycles += 1;
    }
    assert_eq!(res, RuckigResult::Finished);
    let elapsed = cycles as f64 * dt;
    assert_float_eq!(elapsed, output.trajectory.get_duration(), abs <= 2.0 * dt);
}

#[test]
fn test_solve_cub_with_a_zero_constant_term() {
    // x^3 - 3x^2 + 2x = x (x - 1)(x - 2). With d = 0 C++ Ruckig's
    // roots::solve_cubic takes x = 0 and solves x^2 - 3x + 2 = 0, the
    // coefficients shifted down; the port solved x^2 - 3x + d = 0 with the
    // old d, and returned {0, 3}.
    let roots: Vec<f64> = rsruckig::roots::solve_cub(1.0, -3.0, 2.0, 0.0)
        .into_iter()
        .collect();
    assert_eq!(roots, [0.0, 1.0, 2.0]);

    // Its linear case, 2x^2 - 4x = 2x (x - 2): the port returned {0}.
    let roots: Vec<f64> = rsruckig::roots::solve_cub(0.0, 2.0, -4.0, 0.0)
        .into_iter()
        .collect();
    assert_eq!(roots, [0.0, 2.0]);
}

#[test]
fn test_solve_cub_three_real_roots_as_cpp() {
    // Three real roots: C++ Ruckig takes the angle as atan(y/x) (+ pi for
    // x < 0), the port as atan2(y, x); equal in exact arithmetic, here an ulp
    // apart in the middle root. Expected roots: C++ Ruckig 0.19.4's
    // roots::solve_cubic compiled with g++ on glibc, as the Rust build.
    let roots: Vec<f64> =
        rsruckig::roots::solve_cub(1.0, -17.464806435260094, 75.99874721051052, -4.2821278752708745)
            .into_iter()
            .collect();
    assert_eq!(roots, [0.05709129450392858, 7.836552326153905, 9.5711628146022587]);
}

#[test]
fn test_poly_eval_next_to_one() {
    // C++ Ruckig's roots::poly_eval sums the coefficients only at x == 1.0
    // exactly; the port did so for |x - 1| < DBL_EPSILON. At x = 1 - 2^-53
    // the general evaluation of (x - 1)^3's coefficients gives 2^-53, the sum 0.
    let p: arrayvec::ArrayVec<f64, 4> = [1.0, -3.0, 3.0, -1.0].into_iter().collect();
    let x = 1.0 - f64::EPSILON / 2.0;
    assert_eq!(rsruckig::roots::poly_eval(&p, x), f64::EPSILON / 2.0);
    assert_eq!(rsruckig::roots::poly_eval(&p, 1.0), 0.0);
}

#[test]
fn test_step2_failure_is_a_synchronization_error() {
    // C++ Ruckig returns ErrorSynchronizationCalculation when step 2 finds no
    // profile for a DoF; the port returned ErrorExecutionTimeCalculation, the
    // code of step 1. Three rest-to-rest DoFs, phase synchronisation, not
    // collinear; C++ Ruckig 0.19.4 fails step 2 for DoF 2 on this input.
    let mut input = InputParameter::<3>::new(None);
    input.synchronization = Synchronization::Phase;
    input.target_position = DataArrayOrVec::Stack([173.3985511435795, -10.164076494485913, -16.99089134758714]);
    input.max_velocity = DataArrayOrVec::Stack([0.037564289748138364, 1.972109650321082, 1.7571560301775142]);
    input.max_acceleration = DataArrayOrVec::Stack([4366.711651797925, 183.49031770828947, 33.43799463703024]);
    input.min_acceleration = Some(DataArrayOrVec::Stack([-2488.816523939997, -1.1325933606420886, -36.85325869019521]));
    input.max_jerk = DataArrayOrVec::Stack([14412.329460747587, 45630.57859749189, 216165.10759272304]);
    let mut otg = Ruckig::<3, IgnoreErrorHandler>::new(None, 0.001);
    let mut traj = Trajectory::<3>::new(None);
    let result = otg.calculate(&input, &mut traj);
    assert!(
        matches!(result, Ok(RuckigResult::ErrorSynchronizationCalculation)),
        "{result:?}"
    );
}

#[test]
fn test_sub_epsilon_duration_is_synchronized() {
    // C++ Ruckig skips synchronisation only for a duration of exactly 0; the
    // port skipped it below DBL_EPSILON. Two DoFs whose acceleration only
    // needs to reach zero, in 1e-17 s and 1e-18 s; durations and profiles
    // are C++ Ruckig 0.19.4's (step 2 keeps DoF 1's own 1e-18 s profile).
    let j = 1e7;
    let a = [1e-10, 1e-11];
    let mut input = InputParameter::<2>::new(None);
    input.current_acceleration = DataArrayOrVec::Stack(a);
    input.target_position = DataArrayOrVec::Stack(a.map(|a| a * a * a / (3.0 * j * j)));
    input.target_velocity = DataArrayOrVec::Stack(a.map(|a| a * a / (2.0 * j)));
    input.max_velocity = DataArrayOrVec::Stack([1.0, 1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0, 1.0]);
    input.max_jerk = DataArrayOrVec::Stack([j, j]);
    let mut otg = Ruckig::<2, ThrowErrorHandler>::new(None, 0.001);
    let mut traj = Trajectory::<2>::new(None);
    otg.calculate(&input, &mut traj).unwrap();
    assert_eq!(traj.get_duration(), 1e-17);
    let profiles = &traj.get_profiles()[0];
    assert_float_eq!(profiles[0].t[2], 1e-17, r2nd <= 1e-12);
    assert_float_eq!(profiles[1].t[2], 1e-18, r2nd <= 1e-12);
}

#[test]
fn test_first_time_at_position_crossed_twice_in_one_phase() {
    // Starting backwards at -1, the move brakes at the acceleration limit for
    // about 2 s (phase 1, zero jerk), passing -0.25 on the way back (t ~ 0.29)
    // and again on the way forward (t ~ 1.71). C++ Ruckig's roots::Set sorts
    // the cubic's roots before they are read, so the first crossing is found;
    // the port read them unsorted and returned the second. Expected time:
    // C++ Ruckig 0.19.4's (Python package ruckig==0.19.4).
    let mut input = InputParameter::<1>::new(None);
    input.current_velocity = DataArrayOrVec::Stack([-1.0]);
    input.target_position = DataArrayOrVec::Stack([1.0]);
    input.max_velocity = DataArrayOrVec::Stack([1.0]);
    input.max_acceleration = DataArrayOrVec::Stack([1.0]);
    input.max_jerk = DataArrayOrVec::Stack([1000.0]);

    let mut otg = Ruckig::<1, ThrowErrorHandler>::new(None, 0.001);
    let mut traj = Trajectory::<1>::new(None);
    otg.calculate(&input, &mut traj).unwrap();
    let time = traj.get_first_time_at_position(0, -0.25).unwrap();
    assert_float_eq!(time, 0.29268652409927465, abs <= 1e-12);
}

#[test]
fn test_step2_profiles_keep_the_jerk_limit() {
    // Step 2 profiles whose jerk jf is solved for (ACC0_ACC1, the
    // rest-to-rest NONE profile, the 3-step UZU profile) are checked as C++
    // Ruckig checks them, check_with_timing(tf, jf, ..., jMax): |jf| below the
    // jerk limit. A rest-to-rest move of 1 with jerk limit 1 needs 3.17 s;
    // asked to take 3 s, the NONE profile needs jf = 32/27 > 1, which the port
    // returned. Step 2 now finds no profile, as no profile within the limits
    // takes 3 s.
    use rsruckig::position_third_step2::PositionThirdOrderStep2;
    use rsruckig::profile::Profile;
    let (tf, j_max) = (3.0, 1.0);
    let mut step2 =
        PositionThirdOrderStep2::new(tf, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 10.0, -10.0, 10.0, -10.0, j_max);
    let mut profile = Profile::default();
    profile.set_boundary(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    let found = step2.get_profile(&mut profile);
    assert!(
        !found || (0..7).all(|i| profile.t[i] == 0.0 || profile.j[i].abs() <= j_max + 1e-12),
        "found {found}: j {:?} t {:?}",
        profile.j,
        profile.t
    );
}

#[test]
fn test_step2_uzu_profile() {
    // 3 DoF, time synchronised: DoF 1 has a0 = -af and v0 = vf, so the cubic
    // of step 2's 3-step UZU profile has a zero constant term. The port
    // checked that profile with the jerk limit instead of its own jerk jf
    // (C++ Ruckig: check_with_timing(tf, jf, ..., jMax)) and failed step 2.
    // Inputs from a differential test against C++ (random 3-DoF states);
    // expected duration and DoF 0's profile are C++ Ruckig 0.19.4's.
    // (p0, v0, a0, pf, vf, af, v_max, v_min, a_max, a_min, j_max) per DoF
    let dofs: [[f64; 11]; 3] = [
        [
            0.0,
            9.546081433000555,
            -4.374974668969038,
            -2.9544588945012147e-05,
            9.546081433000555,
            4.374974668969038,
            108.86518595799528,
            -0.05831162197231956,
            122.54228508930053,
            -122.54228508930053,
            94698.93239213436,
        ],
        [
            0.0,
            47.79174301084321,
            -123.32537516818434,
            4.17119824234245,
            47.79174301084321,
            10.367363136480668,
            534.445092064759,
            -0.026374538995216138,
            10.367363136480668,
            -3201.841567288944,
            622.5532633348021,
        ],
        [
            0.0,
            -0.002608706532926195,
            3506.6775748524033,
            1.8032944003066386e-05,
            -0.002608706532926195,
            -21.513732800841602,
            0.026107724570856614,
            -0.026107724570856614,
            57363.507563508385,
            -21.513732800841602,
            6795461.366525948,
        ],
    ];
    let mut input = InputParameter::<3>::new(None);
    input.min_velocity = Some(DataArrayOrVec::Stack([0.0; 3]));
    input.min_acceleration = Some(DataArrayOrVec::Stack([0.0; 3]));
    for (d, x) in dofs.iter().enumerate() {
        input.current_position[d] = x[0];
        input.current_velocity[d] = x[1];
        input.current_acceleration[d] = x[2];
        input.target_position[d] = x[3];
        input.target_velocity[d] = x[4];
        input.target_acceleration[d] = x[5];
        input.max_velocity[d] = x[6];
        input.min_velocity.as_mut().unwrap()[d] = x[7];
        input.max_acceleration[d] = x[8];
        input.min_acceleration.as_mut().unwrap()[d] = x[9];
        input.max_jerk[d] = x[10];
    }
    let mut otg = Ruckig::<3, ThrowErrorHandler>::new(None, 0.001);
    let mut traj = Trajectory::<3>::new(None);
    let result = otg.calculate(&input, &mut traj);
    assert!(result.is_ok(), "{result:?}");
    assert_float_eq!(traj.get_duration(), 4309.568938701129, r2nd <= 1e-12);
    let t = traj.get_profiles()[0][0].t;
    let expected = [4.36689820438187, 4300.835142292365, 0.0, 0.0, 0.0, 0.0, 4.366898204381869];
    assert!(almost_equal_vecs(&t, &expected, 1e-9), "{t:?}");
}
