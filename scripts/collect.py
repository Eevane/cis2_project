#!/usr/bin/env python

# Author: Junxiang Wang
# Date: 2024-04-12

# (C) Copyright 2024-2025 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

""" Multilateral teleoperation two consoles (four arms) - python Multitask """
""" Strong connected structure """

import argparse
import time
from enum import Enum
import math
import numpy
import sys
import time
from scipy.spatial.transform import Rotation as R
import cisstVectorPython as cisstVector
import csv

class teleoperation:
    class State(Enum):
        ALIGNING = 1
        CLUTCHED = 2
        FOLLOWING = 3

    def __init__(self, clutch_topic, run_period, align_mtm, operator_present_topic="", alpha = 0.5, **kwargs):
        num = len(kwargs)
        assert num == 6 and {'PSML', 'PSMR'} <= kwargs.keys(), "Wrong configuration, make sure the system has six arms."
        print("Detect six arms, initializing the full system.")
        self.master1_l = kwargs['MTML_expert']
        self.master2_l = kwargs['MTML_novice']
        self.master1_r = kwargs['MTMR_expert']
        self.master2_r = kwargs['MTMR_novice']
        self.puppet_l = kwargs['PSML']
        self.puppet_r = kwargs['PSMR']

        print(f"running at frequency of {1/run_period}")
        self.run_period = run_period

        # dominance factor
        assert 0.0 <= alpha <= 1.0 , "Dominance factor should be at [0, 1]."
        self.alpha = alpha

        self.scale = 0.2
        self.velocity_scale = 0.2

        self.gripper_max = 60 * math.pi / 180
        self.gripper_zero = 0.0 # Set to e.g. 20 degrees if gripper cannot close past zero
        self.jaw_min = -20 * math.pi / 180
        self.jaw_max = 80 * math.pi / 180
        self.jaw_rate = 2 * math.pi

        self.can_align_mtm = align_mtm

        # slowly eliminate alignment offset if we can align mtm,
        # otherwise maintain fixed initial alignment offset
        self.align_rate = 0.25 * math.pi if self.can_align_mtm else 0.0

        # don't require alignment before beginning teleop if mtm wrist can't be actuated
        self.operator_orientation_tolerance = 5 * math.pi / 180 if self.can_align_mtm else math.pi
        self.operator_gripper_threshold = 5 * math.pi / 180
        self.operator_roll_threshold = 3 * math.pi / 180

        self.gripper_to_jaw_scale = self.jaw_max / (self.gripper_max - self.gripper_zero)
        self.gripper_to_jaw_offset = -self.gripper_zero * self.gripper_to_jaw_scale

        self.operator_is_active = False
        self.operator_is_active_onleft = False
        self.operator_is_active_onright = False
        if operator_present_topic:
            self.operator_is_present = False
            self.operator_button = operator_present_topic
        else:
            self.operator_is_present = True # if not given, then always assume present

        self.clutch_pressed = False
        self.clutch_button = clutch_topic

        # for plotting
        self.count = 0
        self.time_data = []
        self.y_data_l = []
        self.y_data_l_expected = []
        self.y_data_r = []
        self.y_data_r_expected = []
        self.m1_l_force = []
        self.m2_l_force = []
        self.m1_r_force = []
        self.m2_r_force = []
        self.puppet_l_force = []
        self.puppet_r_force = []

        # control law gain
        self.force_gain = 0.35
        self.velocity_gain = 1.1

        """for recording"""
        self.start_time = time.monotonic()
        self.recording_enabled = False
        self.record_size = 0

        self.output_csv_path = f"/home/xle6/dvrk_teleop_data/Aug_17/{self.start_time:.6f}-dVRK-Last-Train-joint_data.csv"
        self.csv_file = open(self.output_csv_path, "a", newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.header_written = os.path.getsize(self.output_csv_path) > 0

    def set_velocity_goal(self, v, base=1.10, max_gain=1.27, threshold=0.25):
        norm = numpy.linalg.norm(v)
        # print(f"v: {norm}")
        if norm < threshold:
            gain = max_gain  
        else:
            gain = max_gain * numpy.exp(-12 * ( norm - threshold))
            gain = numpy.maximum(base, gain)   
        return v * gain

    def set_vctFrm3(self, rotation=None, translation=None):
        vctFrm3 = cisstVector.vctFrm3()
        if rotation is not None:
            assert all([isinstance(rotation, numpy.ndarray), rotation.shape == (3,3)])
            vctFrm3.SetRotation(rotation)
        
        if translation is not None:
            assert all([isinstance(translation, numpy.ndarray), translation.shape == (3,)])  #??
            vctFrm3.SetTranslation(translation)
        return vctFrm3


    def GetRotAngle(self, Rot):
        r = R.from_matrix(Rot)
        rotvec = r.as_rotvec()

        theta = numpy.linalg.norm(rotvec)
        axis = rotvec / theta
        return theta, axis


    def GetRotMatrix(self, axis, theta):
        # Ensure the axis is a unit vector
        axis = axis / numpy.linalg.norm(axis)

        rotvec = axis * theta
        r = R.from_rotvec(rotvec)
        rot_mat = r.as_matrix()
        return rot_mat
    
    # average rotation by quaternion
    def average_rotation(self, rotation1, rotation2, alpha=0.5):
        # transfrom into scipy.rotation type
        rot_mats = numpy.array([rotation1, rotation2])
        rots = R.from_matrix(rot_mats)

        weights = [alpha, 1-alpha]
        mean_rot = rots.mean(weights=weights)
        mean_rot_mat = mean_rot.as_matrix()
        return mean_rot_mat

    # callback for operator pedal/button
    def on_operator_present(self):
        present = self.operator_button.GetButton()
        self.operator_is_present = present
        if not present:
            self.operator_is_active = False
        return self.operator_is_present

    # callback for clutch pedal/button
    def on_clutch(self):
        self.clutch_pressed = self.clutch_button.GetButton()
        return self.clutch_pressed

    # compute relative orientation of any mtm and any psm
    def alignment_offset(self, arm1, arm2):
        _, master_rotation = arm1.measured_cp()
        _, puppet_rotation = arm2.setpoint_cp()
        alignment_offset = numpy.linalg.inv(master_rotation) @ puppet_rotation
        return alignment_offset

    # def alignment_offset_master1(self):
    #     _, master1_rotation = self.master1.measured_cp()
    #     _, puppet_rotation = self.puppet.setpoint_cp()
    #     alignment_offset = numpy.linalg.inv(master1_rotation) @ puppet_rotation
    #     return alignment_offset
    
    # # compute relative orientation of mtm2 and psm
    # def alignment_offset_master2(self):
    #     _, master2_rotation = self.master2.measured_cp()
    #     _, puppet_rotation = self.puppet.setpoint_cp()
    #     alignment_offset = numpy.linalg.inv(master2_rotation) @ puppet_rotation
    #     return alignment_offset
    
    # ############################################################################
    # # compute relative orientation of mtm1 and mtm2
    # def alignment_offset_master1_to_master2(self):
    #     _, master1_rotation = self.master1.measured_cp()
    #     _, master2_rotation = self.master2.measured_cp()
    #     alignment_offset = numpy.linalg.inv(master1_rotation) @ master2_rotation
    #     return alignment_offset
    # ############################################################################

    # set relative origins for clutching and alignment offset
    def update_initial_state(self):
        # master1_l
        master1_l_measured_trans, master1_l_measured_rot = self.master1_l.measured_cp()
        self.master1_l_cartesian_initial = self.set_vctFrm3(rotation=master1_l_measured_rot, translation=master1_l_measured_trans)
        # master1_r
        master1_r_measured_trans, master1_r_measured_rot = self.master1_r.measured_cp()
        self.master1_r_cartesian_initial = self.set_vctFrm3(rotation=master1_r_measured_rot, translation=master1_r_measured_trans)
        # master2_l
        master2_l_measured_trans, master2_l_measured_rot = self.master2_l.measured_cp()
        self.master2_l_cartesian_initial = self.set_vctFrm3(rotation=master2_l_measured_rot, translation=master2_l_measured_trans)
        # master2_r
        master2_r_measured_trans, master2_r_measured_rot = self.master2_r.measured_cp()
        self.master2_r_cartesian_initial = self.set_vctFrm3(rotation=master2_r_measured_rot, translation=master2_r_measured_trans)
        # puppet_l
        puppet_l_measured_trans, puppet_l_measured_rot = self.puppet_l.measured_cp()
        self.puppet_l_cartesian_initial = self.set_vctFrm3(rotation=puppet_l_measured_rot, translation=puppet_l_measured_trans)
        # puppet_r
        puppet_r_measured_trans, puppet_r_measured_rot = self.puppet_r.measured_cp()
        self.puppet_r_cartesian_initial = self.set_vctFrm3(rotation=puppet_r_measured_rot, translation=puppet_r_measured_trans)

        #----- left arms initial state
        self.alignment_offset_initial_master1_l = self.alignment_offset(self.master1_l, self.puppet_l)
        self.alignment_offset_initial_master2_l = self.alignment_offset(self.master2_l, self.puppet_l)
        self.alignment_offset_initial_masters_l = self.alignment_offset(self.master1_l, self.master2_l)

        self.master1_l_offset_angle, self.master1_l_offset_axis = self.GetRotAngle(self.alignment_offset_initial_master1_l)
        self.master2_l_offset_angle, self.master2_l_offset_axis = self.GetRotAngle(self.alignment_offset_initial_master2_l)
        self.masters_l_offset_angle, self.masters_l_offset_axis = self.GetRotAngle(self.alignment_offset_initial_masters_l)
        #-----

        #----- right arms initial state
        self.alignment_offset_initial_master1_r = self.alignment_offset(self.master1_r, self.puppet_r)
        self.alignment_offset_initial_master2_r = self.alignment_offset(self.master2_r, self.puppet_r)
        self.alignment_offset_initial_masters_r = self.alignment_offset(self.master1_r, self.master2_r)

        self.master1_r_offset_angle, self.master1_r_offset_axis = self.GetRotAngle(self.alignment_offset_initial_master1_r)
        self.master2_r_offset_angle, self.master2_r_offset_axis = self.GetRotAngle(self.alignment_offset_initial_master2_r)
        self.masters_r_offset_angle, self.masters_r_offset_axis = self.GetRotAngle(self.alignment_offset_initial_masters_r)
        #-----

    def gripper_to_jaw(self, gripper_angle):
        jaw_angle = self.gripper_to_jaw_scale * gripper_angle + self.gripper_to_jaw_offset
        # make sure we don't set goal past joint limits
        return max(jaw_angle, self.jaw_min)

    def jaw_to_gripper(self, jaw_angle):
        return (jaw_angle - self.gripper_to_jaw_offset) / self.gripper_to_jaw_scale

    def enter_aligning(self):
        self.current_state = teleoperation.State.ALIGNING
        self.last_align = None
        self.last_operator_prompt = time.perf_counter()

        self.master1_l.arm.use_gravity_compensation(True)
        self.master1_r.arm.use_gravity_compensation(True)
        self.master2_l.arm.use_gravity_compensation(True)
        self.master2_r.arm.use_gravity_compensation(True)
        self.puppet_l.arm.hold()
        self.puppet_r.arm.hold()

        # reset operator activity data in case operator is inactive
        self.operator_roll_min_master1_l = math.pi * 100
        self.operator_roll_max_master1_l = -math.pi * 100
        self.operator_gripper_min_master1_l = math.pi * 100
        self.operator_gripper_max_master1_l = -math.pi * 100

        self.operator_roll_min_master2_l = math.pi * 100
        self.operator_roll_max_master2_l = -math.pi * 100
        self.operator_gripper_min_master2_l = math.pi * 100
        self.operator_gripper_max_master2_l = -math.pi * 100

        self.operator_roll_min_master1_r = math.pi * 100
        self.operator_roll_max_master1_r = -math.pi * 100
        self.operator_gripper_min_master1_r = math.pi * 100
        self.operator_gripper_max_master1_r = -math.pi * 100

        self.operator_roll_min_master2_r = math.pi * 100
        self.operator_roll_max_master2_r = -math.pi * 100
        self.operator_gripper_min_master2_r = math.pi * 100
        self.operator_gripper_max_master2_r = -math.pi * 100

    def transition_aligning(self):
        if self.operator_is_active and self.on_clutch():
            self.enter_clutched()
            return

        #----- check aligning left
        master1_l_alignment_offset = self.alignment_offset(self.master1_l, self.puppet_l)
        master1_l_orientation_error, _ = self.GetRotAngle(master1_l_alignment_offset)
        master2_l_alignment_offset = self.alignment_offset(self.master2_l, self.puppet_l)
        master2_l_orientation_error, _ = self.GetRotAngle(master2_l_alignment_offset)

        master1_l_aligned = master1_l_orientation_error <= self.operator_orientation_tolerance
        master2_l_aligned = master2_l_orientation_error <= self.operator_orientation_tolerance
        aligned_left = master1_l_aligned and master2_l_aligned
        #-----

        #----- check aligning right
        master1_r_alignment_offset = self.alignment_offset(self.master1_r, self.puppet_r)
        master1_r_orientation_error, _ = self.GetRotAngle(master1_r_alignment_offset)
        master2_r_alignment_offset = self.alignment_offset(self.master2_r, self.puppet_r)
        master2_r_orientation_error, _ = self.GetRotAngle(master2_r_alignment_offset)

        master1_r_aligned = master1_r_orientation_error <= self.operator_orientation_tolerance
        master2_r_aligned = master2_r_orientation_error <= self.operator_orientation_tolerance
        aligned_right = master1_r_aligned and master2_r_aligned
        #-----
        
        # print(f"If master1 is aligned: {master1_aligned}")
        # print(f"If master2 is aligned: {master2_aligned}")
        if aligned_left and aligned_right and self.operator_is_active:
            self.enter_following()

    def run_aligning(self):
        master1_l_alignment_offset = self.alignment_offset(self.master1_l, self.puppet_l)
        master1_l_orientation_error, _ = self.GetRotAngle(master1_l_alignment_offset)
        master2_l_alignment_offset = self.alignment_offset(self.master2_l, self.puppet_l)
        master2_l_orientation_error, _ = self.GetRotAngle(master2_l_alignment_offset)

        master1_r_alignment_offset = self.alignment_offset(self.master1_r, self.puppet_r)
        master1_r_orientation_error, _ = self.GetRotAngle(master1_r_alignment_offset)
        master2_r_alignment_offset = self.alignment_offset(self.master2_r, self.puppet_r)
        master2_r_orientation_error, _ = self.GetRotAngle(master2_r_alignment_offset)

        # if operator is inactive, use gripper or roll activity to detect when the user is ready
        if self.operator_is_present:
            #----- left-hand side
            master1_gripper = self.master1_l.gripper_measured_js()
            master2_gripper = self.master2_l.gripper_measured_js()

            self.operator_gripper_max_master1_l = max(master1_gripper, self.operator_gripper_max_master1_l)
            self.operator_gripper_min_master1_l = min(master2_gripper, self.operator_gripper_min_master1_l)
            master1_gripper_range = self.operator_gripper_max_master1_l - self.operator_gripper_min_master1_l

            self.operator_gripper_max_master2_l = max(master2_gripper, self.operator_gripper_max_master2_l)
            self.operator_gripper_min_master2_l = min(master2_gripper, self.operator_gripper_min_master2_l)
            master2_gripper_range = self.operator_gripper_max_master2_l - self.operator_gripper_min_master2_l
            
            if master1_gripper_range >= self.operator_gripper_threshold or master2_gripper_range >= self.operator_gripper_threshold:
                self.operator_is_active_onleft = True
            #-----

            #----- right-hand side
            master1_gripper = self.master1_r.gripper_measured_js()
            master2_gripper = self.master2_r.gripper_measured_js()

            self.operator_gripper_max_master1_r = max(master1_gripper, self.operator_gripper_max_master1_r)
            self.operator_gripper_min_master1_r = min(master1_gripper, self.operator_gripper_min_master1_r)
            master1_gripper_range = self.operator_gripper_max_master1_r - self.operator_gripper_min_master1_r

            self.operator_gripper_max_master2_r = max(master2_gripper, self.operator_gripper_max_master2_r)
            self.operator_gripper_min_master2_r = min(master2_gripper, self.operator_gripper_min_master2_r)
            master2_gripper_range = self.operator_gripper_max_master2_r - self.operator_gripper_min_master2_r

            if master1_gripper_range >= self.operator_gripper_threshold or master2_gripper_range >= self.operator_gripper_threshold:
                self.operator_is_active_onright = True
            #-----
                
            # determine amount of roll around z axis by rotation of y-axis
            #----- left-hand side
            _, master1_rotation = self.master1_l.measured_cp()
            _, master2_rotation = self.master2_l.measured_cp()
            _, puppet_rotation = self.puppet_l.measured_cp()

            master1_y_axis = numpy.array([master1_rotation[0,1], master1_rotation[1,1], master1_rotation[2,1]])
            master2_y_axis = numpy.array([master2_rotation[0,1], master2_rotation[1,1], master2_rotation[2,1]])
            puppet_y_axis = numpy.array([puppet_rotation[0,1], puppet_rotation[1,1], puppet_rotation[2,1]])
            roll_master1 = math.acos(numpy.dot(puppet_y_axis, master1_y_axis))
            roll_master2 = math.acos(numpy.dot(puppet_y_axis, master2_y_axis))

            self.operator_roll_max_master1_l = max(roll_master1, self.operator_roll_max_master1_l)
            self.operator_roll_min_master1_l = min(roll_master1, self.operator_roll_min_master1_l)
            master1_roll_range = self.operator_roll_max_master1_l - self.operator_roll_min_master1_l

            self.operator_roll_max_master2_l = max(roll_master2, self.operator_roll_max_master2_l)
            self.operator_roll_min_master2_l = min(roll_master2, self.operator_roll_min_master2_l)
            master2_roll_range = self.operator_roll_max_master2_l - self.operator_roll_min_master2_l

            if master1_roll_range >= self.operator_roll_threshold or master2_roll_range >= self.operator_roll_threshold:
                self.operator_is_active_onleft = True
            #-----

            #----- right-hand side
            _, master1_rotation = self.master1_r.measured_cp()
            _, master2_rotation = self.master2_r.measured_cp()
            _, puppet_rotation = self.puppet_r.measured_cp()

            master1_y_axis = numpy.array([master1_rotation[0,1], master1_rotation[1,1], master1_rotation[2,1]])
            master2_y_axis = numpy.array([master2_rotation[0,1], master2_rotation[1,1], master2_rotation[2,1]])
            puppet_y_axis = numpy.array([puppet_rotation[0,1], puppet_rotation[1,1], puppet_rotation[2,1]])
            roll_master1 = math.acos(numpy.dot(puppet_y_axis, master1_y_axis))
            roll_master2 = math.acos(numpy.dot(puppet_y_axis, master2_y_axis))

            self.operator_roll_max_master1_l = max(roll_master1, self.operator_roll_max_master1_l)
            self.operator_roll_min_master1_l = min(roll_master1, self.operator_roll_min_master1_l)
            master1_roll_range = self.operator_roll_max_master1_l - self.operator_roll_min_master1_l

            self.operator_roll_max_master2_l = max(roll_master2, self.operator_roll_max_master2_l)
            self.operator_roll_min_master2_l = min(roll_master2, self.operator_roll_min_master2_l)
            master2_roll_range = self.operator_roll_max_master2_l - self.operator_roll_min_master2_l

            if master1_roll_range >= self.operator_roll_threshold or master2_roll_range >= self.operator_roll_threshold:
               self.operator_is_active_onright = True
            #-----

            if self.operator_is_active_onleft and self.operator_is_active_onright:
                self.operator_is_active =  True
            
        # periodically send move_cp to MTM to align with PSM
        master1_l_aligned = master1_l_orientation_error <= self.operator_orientation_tolerance
        master2_l_aligned = master2_l_orientation_error <= self.operator_orientation_tolerance
        master1_r_aligned = master1_r_orientation_error <= self.operator_orientation_tolerance
        master2_r_aligned = master2_r_orientation_error <= self.operator_orientation_tolerance

        now = time.perf_counter()
        if not self.last_align or now - self.last_align > 4.0:
            # align left arms
            _, goal_rotation = self.puppet_l.setpoint_cp()
            master1_goal_translation, _ = self.master1_l.setpoint_cp()
            master2_goal_translation, _ = self.master2_l.setpoint_cp()

            master1_move_goal = self.set_vctFrm3(rotation=goal_rotation, translation=master1_goal_translation)
            master2_move_goal = self.set_vctFrm3(rotation=goal_rotation, translation=master2_goal_translation)

            self.master1_l.move_cp(master1_move_goal)
            self.master2_l.move_cp(master2_move_goal)

            # align right arms
            _, goal_rotation = self.puppet_r.setpoint_cp()
            master1_goal_translation, _ = self.master1_r.setpoint_cp()
            master2_goal_translation, _ = self.master2_r.setpoint_cp()

            master1_move_goal = self.set_vctFrm3(rotation=goal_rotation, translation=master1_goal_translation)
            master2_move_goal = self.set_vctFrm3(rotation=goal_rotation, translation=master2_goal_translation)

            self.master1_r.move_cp(master1_move_goal)
            self.master2_r.move_cp(master2_move_goal)
            self.last_align = now

        # periodically notify operator if un-aligned or operator is inactive
        if self.operator_is_present and now - self.last_operator_prompt > 4.0:
            self.last_operator_prompt = now
            if not master1_l_aligned:
                print(f'Unable to align master ({self.master1_l.name}), angle error is {master1_l_orientation_error * 180 / math.pi} (deg)')
            if not master2_l_aligned:
                print(f'Unable to align master ({self.master2_l.name}), angle error is {master2_l_orientation_error * 180 / math.pi} (deg)')
            if not master1_r_aligned:
                print(f'Unable to align master ({self.master1_r.name}), angle error is {master1_r_orientation_error * 180 / math.pi} (deg)')
            if not master2_r_aligned:
                print(f'Unable to align master ({self.master2_r.name}), angle error is {master2_r_orientation_error * 180 / math.pi} (deg)')
            elif not self.operator_is_active:
                print(f'To begin teleop, pinch/twist master ({self.master1_l.name}) or ({self.master2_l.name}) gripper a bit')

    def enter_clutched(self):
        self.current_state = teleoperation.State.CLUTCHED
        print("enter clutch!")
        # let MTM position move freely, but lock orientation
        wrench = numpy.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.master1_l.body_servo_cf(wrench)
        self.master2_l.body_servo_cf(wrench)

        self.master1_l.lock_orientation()
        self.master2_l.lock_orientation()
        self.puppet_l.arm.hold()

        self.master1_r.body_servo_cf(wrench)
        self.master2_r.body_servo_cf(wrench)

        self.master1_r.lock_orientation()
        self.master2_r.lock_orientation()
        self.puppet_r.arm.hold()

    def transition_clutched(self):
        if not self.on_clutch() or not self.on_operator_present():
            self.enter_aligning()

    def run_clutched(self):
        pass

    def enter_following(self):
        self.current_state = teleoperation.State.FOLLOWING
        # update MTM/PSM origins position
        self.update_initial_state()

        # set up gripper ghost to rate-limit jaw speed
        jaw_l_setpoint = self.puppet_l.jaw_setpoint_js()
        jaw_r_setpoint = self.puppet_r.jaw_setpoint_js()

        # avoid []
        if len(jaw_l_setpoint) != 1:
            print(f'Unable to get jaw position. Make sure there is an instrument on the puppet ({self.puppet_l.name})')
            self.running = False
        if len(jaw_r_setpoint) != 1:
            print(f'Unable to get jaw position. Make sure there is an instrument on the puppet ({self.puppet_r.name})')
            self.running = False
        self.gripper_l_ghost = self.jaw_to_gripper(jaw_l_setpoint)
        self.gripper_r_ghost = self.jaw_to_gripper(jaw_r_setpoint)

        self.master1_l.arm.use_gravity_compensation(True)
        self.master1_r.arm.use_gravity_compensation(True)
        self.master2_l.arm.use_gravity_compensation(True)
        self.master2_r.arm.use_gravity_compensation(True)

    def transition_following(self):
        if not self.on_operator_present():
            self.enter_aligning()
        elif self.on_clutch():
            self.enter_clutched()

    def run_following(self):
        #-----------------------------------------------------------------------------------------------------
        """ Left arms movement """       
        """
        Forward Process
        """
        # Force channel
        # master1
        master1_l_measured_cf = self.master1_l.body_measured_cf()   # (6,) numpy array
        master1_l_measured_cf[0:3] *= -1.0
        master1_l_measured_cf[3:6] *= 0   # turn off torque

        # master2
        master2_l_measured_cf = self.master2_l.body_measured_cf()   # (6,) numpy array
        master2_l_measured_cf[0:3] *= -1.0
        master2_l_measured_cf[3:6] *= 0   # turn off torque

        # puppet
        puppet_l_measured_cf = self.puppet_l.body_measured_cf()
        puppet_l_measured_cf[0:3] *= -1.0
        puppet_l_measured_cf[3:6] *= 0

        # force input
        gamma = 0.714
        force_l_goal = self.force_gain * (self.alpha * master1_l_measured_cf + (1 - self.alpha) * master2_l_measured_cf + gamma * puppet_l_measured_cf)


        # Position channel
        master1_l_measured_trans, master1_l_measured_rot = self.master1_l.measured_cp()
        master2_l_measured_trans, master2_l_measured_rot = self.master2_l.measured_cp()
        master1_l_initial_trans = self.master1_l_cartesian_initial.GetTranslation()   ####### Reference or copy #########
        master2_l_initial_trans = self.master2_l_cartesian_initial.GetTranslation()
        puppet_l_initial_trans = self.puppet_l_cartesian_initial.GetTranslation()


        # set translation of psm
        master1_l_translation = master1_l_measured_trans - master1_l_initial_trans
        master2_l_translation = master2_l_measured_trans - master2_l_initial_trans
        master1_l_translation *= self.scale
        master2_l_translation *= self.scale
        master_l_total_translation = self.alpha * master1_l_translation + (1 - self.alpha) * master2_l_translation
        puppet_l_position = master_l_total_translation + puppet_l_initial_trans

        # set rotation of psm to match mtm plus alignment offset
        # if we can actuate the MTM, we slowly reduce the alignment offset to zero over time
        max_delta = self.align_rate * self.run_period
        self.master1_l_offset_angle += math.copysign(min(abs(self.master1_l_offset_angle), max_delta), -self.master1_l_offset_angle)
        self.master2_l_offset_angle += math.copysign(min(abs(self.master2_l_offset_angle), max_delta), -self.master2_l_offset_angle)
        
        # rotation offset from master1
        master1_l_alignment_offset = self.GetRotMatrix(self.master1_l_offset_axis, self.master1_l_offset_angle)
        master1_l_rotation_alignment = master1_l_measured_rot @ master1_l_alignment_offset

        # rotation offset from master2
        master2_l_alignment_offset = self.GetRotMatrix(self.master2_l_offset_axis, self.master2_l_offset_angle)
        master2_l_rotation_alignment = master2_l_measured_rot @ master2_l_alignment_offset

        # average rotation
        puppet_l_rotation = self.average_rotation(master1_l_rotation_alignment, master2_l_rotation_alignment, alpha=self.alpha)

        # set cartesian goal of psm
        puppet_l_cartesian_goal = self.set_vctFrm3(rotation=puppet_l_rotation, translation=puppet_l_position)


        # Velocity channel
        master1_l_measured_cv = self.master1_l.measured_cv()   # (6,) numpy array
        master1_l_measured_cv[0:3] *= self.velocity_scale      # scale the linear velocity
        # master1_l_measured_cv[3:6] *= 0.8      # scale down the angular velocity by 0.2

        master2_l_measured_cv = self.master2_l.measured_cv()   # master2
        master2_l_measured_cv[0:3] *= self.velocity_scale     
        # master2_l_measured_cv[3:6] *= 0.8     

        # average velocity
        # puppet_velocity_goal = self.velocity_gain * (master1_measured_cv + master2_measured_cv) / 2.0

        raw_puppet_l_velocity_goal = self.alpha * master1_l_measured_cv + (1 - self.alpha) * master2_l_measured_cv
        puppet_l_velocity_goal = self.set_velocity_goal(v=raw_puppet_l_velocity_goal)
        # Move
        self.puppet_l.servo_cs(puppet_l_cartesian_goal, puppet_l_velocity_goal, force_l_goal)


        ### Jaw/gripper teleop
        current_master1_l_gripper = self.master1_l.gripper_measured_js()
        current_master2_l_gripper = self.master2_l.gripper_measured_js()

        master1_l_ghost_lag = current_master1_l_gripper - self.gripper_l_ghost
        master2_l_ghost_lag = current_master2_l_gripper - self.gripper_l_ghost
        # average gripper lag
        ghost_l_lag = self.alpha * master1_l_ghost_lag + (1 - self.alpha) * master2_l_ghost_lag

        max_delta = self.jaw_rate * self.run_period
        # move ghost at most max_delta towards current gripper
        self.gripper_l_ghost += math.copysign(min(abs(ghost_l_lag), max_delta), ghost_l_lag)
        jaw_l_goal = numpy.array([self.gripper_to_jaw(self.gripper_l_ghost)]).reshape(-1)
        self.puppet_l.jaw_servo_jp(jaw_l_goal)


        """
        Backward Process
        """
        # Position channel
        puppet_l_measured_trans, puppet_l_measured_rot = self.puppet_l.measured_cp()
        puppet_l_translation = puppet_l_measured_trans - puppet_l_initial_trans     ##### should it update puppet initial cartesian after forward process???
        puppet_l_translation /= self.scale

        # set translation of mtm1
        master1_l_position = puppet_l_translation + master1_l_initial_trans
        # set translation of mtm2
        master2_l_position = puppet_l_translation + master2_l_initial_trans

        # set rotation of mtm1
        master1_l_rotation = puppet_l_measured_rot @ numpy.linalg.inv(master1_l_alignment_offset)
        # set rotation of mtm2
        master2_l_rotation = puppet_l_measured_rot @ numpy.linalg.inv(master2_l_alignment_offset)

        # set cartesian goal of mtm1 and mtm2
        master1_l_cartesian_goal = self.set_vctFrm3(rotation=master1_l_rotation, translation=master1_l_position)
        master2_l_cartesian_goal = self.set_vctFrm3(rotation=master2_l_rotation, translation=master2_l_position)


        # Velocity channel
        puppet_l_measured_cv = self.puppet_l.measured_cv()   # (6,) numpy array
        puppet_l_measured_cv[0:3] /= self.velocity_scale      # scale the linear velocity
        # puppet_l_measured_cv[3:6] *= 0.8      # scale down the angular velocity by 0.2

        # set velocity goal
        # master1_velocity_goal = self.velocity_gain * (puppet_measured_cv + master2_measured_cv) / 2.0
        # master2_velocity_goal = self.velocity_gain * (puppet_measured_cv + master1_measured_cv) / 2.0
        master1_l_velocity_goal = self.set_velocity_goal(v=puppet_l_measured_cv)
        master2_l_velocity_goal = self.set_velocity_goal(v=puppet_l_measured_cv)


        # Move
        self.master1_l.servo_cs(master1_l_cartesian_goal, master1_l_velocity_goal, force_l_goal)
        self.master2_l.servo_cs(master2_l_cartesian_goal, master2_l_velocity_goal, force_l_goal)

        #-----------------------------------------------------------------------------------------------------
        """ Right arms movement """       
        """
        Forward Process
        """
        # Force channel
        # master1
        master1_r_measured_cf = self.master1_r.body_measured_cf()   # (6,) numpy array
        master1_r_measured_cf[0:3] *= -1.0
        master1_r_measured_cf[3:6] *= 0   # turn off torque

        # master2
        master2_r_measured_cf = self.master2_r.body_measured_cf()   # (6,) numpy array
        master2_r_measured_cf[0:3] *= -1.0
        master2_r_measured_cf[3:6] *= 0   # turn off torque

        # puppet
        puppet_r_measured_cf = self.puppet_r.body_measured_cf()
        puppet_r_measured_cf[0:3] *= -1.0
        puppet_r_measured_cf[3:6] *= 0

        # force input
        gamma = 0.714
        force_r_goal = self.force_gain * (self.alpha * master1_r_measured_cf + (1 - self.alpha) * master2_r_measured_cf + gamma * puppet_r_measured_cf)


        # Position channel
        master1_r_measured_trans, master1_r_measured_rot = self.master1_r.measured_cp()
        master2_r_measured_trans, master2_r_measured_rot = self.master2_r.measured_cp()
        master1_r_initial_trans = self.master1_r_cartesian_initial.GetTranslation()   ####### Reference or copy #########
        master2_r_initial_trans = self.master2_r_cartesian_initial.GetTranslation()
        puppet_r_initial_trans = self.puppet_r_cartesian_initial.GetTranslation()


        # set translation of psm
        master1_r_translation = master1_r_measured_trans - master1_r_initial_trans
        master2_r_translation = master2_r_measured_trans - master2_r_initial_trans
        master1_r_translation *= self.scale
        master2_r_translation *= self.scale
        master_r_total_translation = (self.alpha * master1_r_translation + (1 - self.alpha) * master2_r_translation)
        puppet_r_position = master_r_total_translation + puppet_r_initial_trans

        # set rotation of psm to match mtm plus alignment offset
        # if we can actuate the MTM, we slowly reduce the alignment offset to zero over time
        max_delta = self.align_rate * self.run_period
        self.master1_r_offset_angle += math.copysign(min(abs(self.master1_r_offset_angle), max_delta), -self.master1_r_offset_angle)
        self.master2_r_offset_angle += math.copysign(min(abs(self.master2_r_offset_angle), max_delta), -self.master2_r_offset_angle)
        
        # rotation offset from master1
        master1_r_alignment_offset = self.GetRotMatrix(self.master1_r_offset_axis, self.master1_r_offset_angle)
        master1_r_rotation_alignment = master1_r_measured_rot @ master1_r_alignment_offset

        # rotation offset from master2
        master2_r_alignment_offset = self.GetRotMatrix(self.master2_r_offset_axis, self.master2_r_offset_angle)
        master2_r_rotation_alignment = master2_r_measured_rot @ master2_r_alignment_offset

        # average rotation
        puppet_r_rotation = self.average_rotation(master1_r_rotation_alignment, master2_r_rotation_alignment, alpha=self.alpha)

        # set cartesian goal of psm
        puppet_r_cartesian_goal = self.set_vctFrm3(rotation=puppet_r_rotation, translation=puppet_r_position)


        # Velocity channel
        master1_r_measured_cv = self.master1_r.measured_cv()   # (6,) numpy array
        master1_r_measured_cv[0:3] *= self.velocity_scale      # scale the linear velocity
        # master1_r_measured_cv[3:6] *= 0.8      # scale down the angular velocity by 0.2

        master2_r_measured_cv = self.master2_r.measured_cv()   # master2
        master2_r_measured_cv[0:3] *= self.velocity_scale     
        # master2_r_measured_cv[3:6] *= 0.8     

        # average velocity
        # puppet_velocity_goal = self.velocity_gain * (master1_measured_cv + master2_measured_cv) / 2.0

        raw_puppet_r_velocity_goal = self.alpha * master1_r_measured_cv + (1 - self.alpha) * master2_r_measured_cv
        puppet_r_velocity_goal = self.set_velocity_goal(v=raw_puppet_r_velocity_goal)
        # Move
        self.puppet_r.servo_cs(puppet_r_cartesian_goal, puppet_r_velocity_goal, force_r_goal)


        ### Jaw/gripper teleop
        current_master1_r_gripper = self.master1_r.gripper_measured_js()
        current_master2_r_gripper = self.master2_r.gripper_measured_js()

        master1_r_ghost_lag = current_master1_r_gripper - self.gripper_r_ghost
        master2_r_ghost_lag = current_master2_r_gripper - self.gripper_r_ghost
        # average gripper lag
        ghost_r_lag = self.alpha * master1_r_ghost_lag + (1 - self.alpha) * master2_r_ghost_lag

        max_delta = self.jaw_rate * self.run_period
        # move ghost at most max_delta towards current gripper
        self.gripper_r_ghost += math.copysign(min(abs(ghost_r_lag), max_delta), ghost_r_lag)
        jaw_r_goal = numpy.array([self.gripper_to_jaw(self.gripper_r_ghost)]).reshape(-1)
        self.puppet_r.jaw_servo_jp(jaw_r_goal)


        """
        Backward Process
        """
        # Position channel
        puppet_r_measured_trans, puppet_r_measured_rot = self.puppet_r.measured_cp()
        puppet_r_translation = puppet_r_measured_trans - puppet_r_initial_trans     ##### should it update puppet initial cartesian after forward process???
        puppet_r_translation /= self.scale

        # set translation of mtm1
        master1_r_position = puppet_r_translation + master1_r_initial_trans
        # set translation of mtm2
        master2_r_position = puppet_r_translation + master2_r_initial_trans

        # set rotation of mtm1
        master1_r_rotation = puppet_r_measured_rot @ numpy.linalg.inv(master1_r_alignment_offset)
        # set rotation of mtm2
        master2_r_rotation = puppet_r_measured_rot @ numpy.linalg.inv(master2_r_alignment_offset)

        # set cartesian goal of mtm1 and mtm2
        master1_r_cartesian_goal = self.set_vctFrm3(rotation=master1_r_rotation, translation=master1_r_position)
        master2_r_cartesian_goal = self.set_vctFrm3(rotation=master2_r_rotation, translation=master2_r_position)


        # Velocity channel
        puppet_r_measured_cv = self.puppet_r.measured_cv()   # (6,) numpy array
        puppet_r_measured_cv[0:3] /= self.velocity_scale      # scale the linear velocity
        # puppet_r_measured_cv[3:6] *= 0.8      # scale down the angular velocity by 0.2

        # set velocity goal
        # master1_velocity_goal = self.velocity_gain * (puppet_measured_cv + master2_measured_cv) / 2.0
        # master2_velocity_goal = self.velocity_gain * (puppet_measured_cv + master1_measured_cv) / 2.0
        # master1_r_velocity_goal = self.set_velocity_goal(v=puppet_r_measured_cv / max(self.alpha,1e-8))
        # master2_r_velocity_goal = self.set_velocity_goal(v=puppet_r_measured_cv / max(self.alpha,1e-8))
        master1_r_velocity_goal = self.set_velocity_goal(v=puppet_r_measured_cv)
        master2_r_velocity_goal = self.set_velocity_goal(v=puppet_r_measured_cv)


        # Move
        self.master1_r.servo_cs(master1_r_cartesian_goal, master1_r_velocity_goal, force_r_goal)
        self.master2_r.servo_cs(master2_r_cartesian_goal, master2_r_velocity_goal, force_r_goal)
        #-----------------------------------------------------------------------------------------------------

        """
        record plotting data
        """
        self.y_data_l.append(puppet_l_measured_trans.copy())
        self.y_data_l_expected.append(puppet_l_position.copy())
        self.m1_l_force.append(master1_l_measured_cf.copy())
        self.m2_l_force.append(master2_l_measured_cf.copy())
        self.puppet_l_force.append(puppet_l_measured_cf.copy())

        self.y_data_r.append(puppet_r_measured_trans.copy())
        self.y_data_r_expected.append(puppet_r_position.copy())
        self.m1_r_force.append(master1_r_measured_cf.copy())
        self.m2_r_force.append(master2_r_measured_cf.copy())
        self.puppet_r_force.append(puppet_r_measured_cf.copy())



        '''For recording'''
        current_time = time.monotonic()
        print(f"recording enabled: {self.recording_enabled}")
        if not self.recording_enabled and float(current_time - self.start_time) >= 30.0:
            print("Start recording joint data")
            self.recording_enabled = True

        if self.recording_enabled and self.record_size >= 200000:
            print("Auto stopping: 100000 series of data acquired.")
            # time.strftime("%Y-%m-%d %H:%M:%S", current_time)
            # time.strftime("%Y-%m-%d %H:%M:%S", self.start_time)
            print(f"start_time: {self.start_time}")
            print(f"end_time: {current_time}")
            self.recording_enabled = False
            self.running = False

        if self.recording_enabled:
            self.record_size += 1
            print("Recording data.")

            timestamp = time.time()

            master1_l_q, master1_l_dq, master1_l_torque = self.master1_l.measured_js()
            master1_l_q = master1_l_q[:6].tolist()
            master1_l_dq = master1_l_dq[:6].tolist()
            master1_l_torque = master1_l_torque[:6].tolist()


            master2_l_q, master2_l_dq, master2_l_torque = self.master2_l.measured_js()
            master2_l_q = master2_l_q[:6].tolist()
            master2_l_dq = master2_l_dq[:6].tolist()
            master2_l_torque = master2_l_torque[:6].tolist()

            puppet_l_q, puppet_l_dq, puppet_l_torque = self.puppet_l.measured_js()
            puppet_l_q = puppet_l_q.tolist()
            puppet_l_dq = puppet_l_dq.tolist()
            puppet_l_torque = puppet_l_torque.tolist()

            master1_r_q, master1_r_dq, master1_r_torque = self.master1_r.measured_js()
            master1_r_q = master1_r_q[:6].tolist()
            master1_r_dq = master1_r_dq[:6].tolist()
            master1_r_torque = master1_r_torque[:6].tolist()


            master2_r_q, master2_r_dq, master2_r_torque = self.master2_r.measured_js()
            master2_r_q = master2_r_q[:6].tolist()
            master2_r_dq = master2_r_dq[:6].tolist()
            master2_r_torque = master2_r_torque[:6].tolist()

            puppet_r_q, puppet_r_dq, puppet_r_torque = self.puppet_r.measured_js()
            puppet_r_q = puppet_r_q.tolist()
            puppet_r_dq = puppet_r_dq.tolist()
            puppet_r_torque = puppet_r_torque.tolist()

            row = [timestamp] + master1_l_q + master1_l_dq + master1_l_torque + master2_l_q + master2_l_dq + master2_l_torque + puppet_l_q + puppet_l_dq + puppet_l_torque + master1_r_q + master1_r_dq + master1_r_torque + master2_r_q + master2_r_dq + master2_r_torque + puppet_r_q + puppet_r_dq + puppet_r_torque

            if not self.header_written:
                headers = ['timestamp'] + \
                        [f'master1_l_q{i}' for i in range(6)] + [f'master1_l_dq{i}' for i in range(6)] + [f'master1_l_tau{i}' for i in range(6)] + \
                        [f'master2_l_q{i}' for i in range(6)] + [f'master2_l_dq{i}' for i in range(6)] + [f'master2_l_tau{i}' for i in range(6)] + \
                        [f'puppet_l_q{i}' for i in range(6)]  + [f'puppet_l_dq{i}' for i in range(6)]  + [f'puppet_l_tau{i}' for i in range(6)] + \
                        [f'master1_r_q{i}' for i in range(6)] + [f'master1_r_dq{i}' for i in range(6)] + [f'master1_r_tau{i}' for i in range(6)] + \
                        [f'master2_r_q{i}' for i in range(6)] + [f'master2_r_dq{i}' for i in range(6)] + [f'master2_r_tau{i}' for i in range(6)] + \
                        [f'puppet_r_q{i}' for i in range(6)]  + [f'puppet_r_dq{i}' for i in range(6)]  + [f'puppet_r_tau{i}' for i in range(6)]
                self.csv_writer.writerow(headers)
                self.header_written = True

            self.csv_writer.writerow(row)
    def home(self):
        print("Homing arms...")
        system.home()
        timeout = 10.0 # seconds
        time.sleep(timeout)
        # if not self.puppet.enable(timeout) or not self.puppet.home(timeout):
        #     print('    ! failed to home {} within {} seconds'.format(self.puppet.name, timeout))
        #     return False

        # if not self.master1.enable(timeout) or not self.master1.home(timeout):
        #     print('    ! failed to home {} within {} seconds'.format(self.master1.name, timeout))
        #     return False
        
        # if not self.master2.enable(timeout) or not self.master2.home(timeout):
        #     print('    ! failed to home {} within {} seconds'.format(self.master2.name, timeout))
        #     return False

        # print("    Homing is complete")
        # return True


    def run(self):
        self.home()
        print("home complete")
        # if not homed_successfully:
        #     print("home not success")
        #     return
        
        puppet_initial_position = numpy.array([-0.00270296, 0.0368143, 0.142947, 1.28645, -0.0889504, 0.174713])
        self.puppet_l.move_jp(puppet_initial_position)
        self.puppet_r.move_jp(puppet_initial_position)
        time.sleep(3)
        
        print("Running teleop at {} Hz".format(int(1/self.run_period)))

        self.enter_aligning()
        print("Initial alignment completed")
        self.running = True

        last_time = time.time()
        try:
            while True:
                # check IO state
                self.on_operator_present()
                self.on_clutch()

                # check if teleop state should transition
                if self.current_state == teleoperation.State.ALIGNING:
                    self.transition_aligning()
                    # print("transition_aligning is finished.")
                elif self.current_state == teleoperation.State.CLUTCHED:
                    self.transition_clutched()
                    # print("transition_clutched is finished.")
                elif self.current_state == teleoperation.State.FOLLOWING:
                    self.transition_following()
                    # print("transition_following is finished.")
                else:
                    raise RuntimeError("Invalid state: {}".format(self.current_state))

                # self.check_arm_state()
                
            
                if not self.running:
                    break

                # run teleop state handler
                if self.current_state == teleoperation.State.ALIGNING:
                    self.run_aligning()
                    # print("run_aligning is finished.")
                elif self.current_state == teleoperation.State.CLUTCHED:
                    self.run_clutched()
                    # print("run_clutched is finished.")
                elif self.current_state == teleoperation.State.FOLLOWING:
                    self.run_following()
                else:
                    raise RuntimeError("Invalid state: {}".format(self.current_state))
                now = time.time()
                to_sleep = self.run_period - (now - last_time)
                if self.count == 200:
                    print(f"########### Teleoperation runs {1/(now - last_time)} times per sec.")
                    self.count = 0
                self.count += 1
                time.sleep(to_sleep) if to_sleep > 0 else None
                last_time = time.time()
                
        except KeyboardInterrupt:
            print("Program stopped!")
            system.power_off()

        # save data
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_11/multi_array.txt', self.y_data_l, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_11/multi_array_exp.txt', self.y_data_l_expected, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_11/PSM_total_force.txt', self.puppet_l_force, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_11/MTML_total_force.txt', self.m1_l_force, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_11/MTMR_total_force.txt', self.m2_l_force, fmt='%f', delimiter=' ', comments='')

        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_11/multi_array.txt', self.y_data_r, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_11/multi_array_exp.txt', self.y_data_r_expected, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_11/PSM_total_force.txt', self.puppet_r_force, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_11/MTML_total_force.txt', self.m1_r_force, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_11/MTMR_total_force.txt', self.m2_r_force, fmt='%f', delimiter=' ', comments='')
        print(f"data.txt saved!")

class ARM:
    def __init__(self, arm, name):
        self.arm = arm
        self.name = name
    
    def measured_js(self):
        measured_js = self.arm.measured_js()
        measured_jf = measured_js.Effort()
        measured_jv = measured_js.Velocity()
        measured_jp = measured_js.Position()
        return measured_jp.copy(), measured_jv.copy(), measured_jf.copy()

    def measured_cp(self):
        measure_cp = self.arm.measured_cp()
        cartesian_position = measure_cp.Position()
        cartesian_orientation = cartesian_position.GetRotation()
        cartesian_translation = cartesian_position.GetTranslation()
        return cartesian_translation.copy(), cartesian_orientation.copy()

    def setpoint_cp(self):
        setpoint_cp = self.arm.setpoint_cp()
        cartesian_position = setpoint_cp.Position()
        cartesian_orientation = cartesian_position.GetRotation()
        cartesian_translation = cartesian_position.GetTranslation()
        return cartesian_translation.copy(), cartesian_orientation.copy()
    
    def measured_cv(self):
        measured_cv = self.arm.measured_cv()
        linear_vel = measured_cv.VelocityLinear()
        angular_vel = measured_cv.VelocityAngular()
        velocity = numpy.hstack((linear_vel, angular_vel))
        return velocity
    
    def body_measured_cf(self):
        body_measured_cf = self.arm.body.measured_cf()
        body_force = body_measured_cf.Force()
        return body_force.copy()

    def gripper_measured_js(self):
        assert self.name.startswith("MTM")
        gripper = self.arm.gripper.measured_js()
        gripper_position = gripper.Position()
        return gripper_position.copy()
    
    def jaw_setpoint_js(self):
        assert self.name.startswith("PSM")
        jaw = self.arm.jaw.setpoint_js()
        jaw_setpoint = jaw.Position()
        return jaw_setpoint.copy()
       
    def jaw_servo_jp(self, goal):
        assert self.name.startswith("PSM")
        assert isinstance(goal, numpy.ndarray)
        arg = self.arm.jaw.servo_jp.GetArgumentPrototype()
        arg.SetGoal(goal)
        self.arm.jaw.servo_jp(arg)
    
    def move_cp(self, goal):
        assert isinstance(goal, cisstVector.vctFrm3)
        arg = self.arm.move_cp.GetArgumentPrototype()
        arg.SetGoal(goal)
        self.arm.move_cp(arg)

    def servo_cs(self, position_goal, velocity_goal, force_goal):
        assert all([isinstance(position_goal, cisstVector.vctFrm3), isinstance(velocity_goal, numpy.ndarray), isinstance(force_goal, numpy.ndarray)])
        arg = self.arm.servo_cs.GetArgumentPrototype()
        arg.SetPositionIsValid(True)
        arg.SetPosition(position_goal)
        arg.SetVelocityIsValid(True)
        arg.SetVelocity(velocity_goal)
        arg.SetForceIsValid(True)
        arg.SetForce(force_goal)
        self.arm.servo_cs(arg)
        
    def body_servo_cf(self, goal):
        assert isinstance(goal, numpy.ndarray)
        arg = self.arm.body.servo_cf.GetArgumentPrototype()
        arg.SetForce(goal)
        self.arm.body.servo_cf(arg)

    def lock_orientation(self):
        _, lock_rot = self.measured_cp()
        self.arm.lock_orientation(lock_rot)

    def move_jp(self, goal):
        arg = self.arm.move_jp.GetArgumentPrototype()
        arg.SetGoal(goal)
        self.arm.move_jp(arg)
        

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alpha', type = float, required = False, default= 0.5,
                         help = 'dominance factor of position channel, between 0 and 1')
    parser.add_argument('-n', '--no-mtm-alignment', action='store_true',
                        help="don't align mtm (useful for using haptic devices as MTM which don't have wrist actuation)")
    parser.add_argument('-i', '--interval', type=float, default=0.0013,
                        help = 'time interval/period to run at - should be as long as system\'s period to prevent timeouts')
    args = parser.parse_args()

    from dvrk_system import *
    mtml1 = ARM(MTML1, 'MTML1')   # create arm instances by arm names defined in the config.json
    mtml2 = ARM(MTML2, 'MTML2')
    psm2 = ARM(PSM2, 'PSM2')

    mtmr1 = ARM(MTMR1, 'MTMR1')
    mtmr2 = ARM(MTMR2, 'MTMR2')
    psm1 = ARM(PSM1, 'PSM1')

    clutch = clutch
    coag = coag

    application = teleoperation(clutch, args.interval,
                                not args.no_mtm_alignment, operator_present_topic = coag, alpha = args.alpha, 
                                MTML_novice=mtml1, MTML_expert=mtml2, MTMR_novice=mtmr1, MTMR_expert=mtmr2, PSML=psm2, PSMR=psm1)
     
    application.run()
