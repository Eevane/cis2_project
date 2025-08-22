#!/usr/bin/env python

# Author: Junxiang Wang
# Date: 2024-04-12

# (C) Copyright 2024-2025 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

""" Multilateral teleoperation single system - Wrapper version """
""" Strong connected structure, model-involved """

import argparse
import time
from enum import Enum
import math
import numpy
import sys
import time
from scipy.spatial.transform import Rotation as R
import cisstVectorPython as cisstVector
import onnxruntime
import threading
from queue import Queue, Empty
from model_merge import ModelMerger

master1_external_f_queue = Queue(maxsize=1)
master2_external_f_queue = Queue(maxsize=1)
puppet_external_f_queue = Queue(maxsize=1)
master1_internal_f_queue = Queue(maxsize=1)
master2_internal_f_queue = Queue(maxsize=1)
puppet_internal_f_queue = Queue(maxsize=1)



internal_torque_record_MTML= []
total_torque_record_MTML = []
internal_torque_record_MTMR = []
total_torque_record_MTMR = []
internal_torque_record_PSM = []
total_torque_record_PSM = []

internal_force_MTML = []
internal_force_MTMR = []
internal_force_PSM = []
cartesian_force_MTML = []
cartesian_force_MTMR = []
cartesian_force_PSM = []

class teleoperation:
    class State(Enum):
        ALIGNING = 1
        CLUTCHED = 2
        FOLLOWING = 3

    def __init__(self, mtm1, mtm2, puppet, clutch_topic, run_period, align_mtm, operator_present_topic="", alpha = 0.5):
        print('Initialzing dvrk_teleoperation for {}, {} and {}'.format(mtm1.name, mtm2.name, puppet.name))
        print(f"running at frequency of {1/run_period}")
        self.run_period = run_period

        self.master1 = mtm1
        self.master2 = mtm2
        self.puppet = puppet

        # dominance factor
        self.alpha = alpha

        self.scale = 0.25
        self.velocity_scale = 0.25

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
        if operator_present_topic:
            self.operator_is_present = False
            self.operator_button = operator_present_topic
        else:
            self.operator_is_present = True # if not given, then always assume present

        self.clutch_pressed = False
        self.clutch_button = clutch_topic

        self.count = 0
        self.time_data = []
        self.y_data_l = []
        self.y_data_l_expected = []

        self.total_force_MTML = []
        self.total_force_MTMR = []
        self.total_force_PSM = []
        #####################################################################################

        # control law gain
        self.force_gain = 0.2
        self.velocity_gain = 1.135



        self.master1_internal_force = 0.0
        self.master2_internal_force = 0.0
        self.puppet_internal_force = 0.0

    def set_velocity_goal(self, v, base=1.10, max_gain=1.18, threshold=0.1):
        norm = numpy.linalg.norm(v)
        # print(f"v: {norm}")
        if norm < threshold:
            gain = max_gain  
        else:
            # gain = base + (max_gain - base) * (threshold / norm)
            gain = max_gain * numpy.exp(-12 * ( norm - threshold))
            gain = numpy.maximum(base, gain)
            # print(f"Interpolated gain: {gain}")
            
        return v * gain
    
    def set_velocity_goal_split(self, v, base=1.05, max_gain=1.19, linear_threshold=0.165, rot_threshold=0.1):
        norm_linear = numpy.linalg.norm(v[0:3])
        norm_rot = numpy.linalg.norm(v[3:6])
        if norm_linear < linear_threshold:
            gain = max_gain
            print(f"low speed")
            
        else:
            # gain = base + (max_gain - base) * (threshold / norm)
            gain = max_gain * numpy.exp(-12 * (norm_linear - linear_threshold))
            gain = numpy.maximum(base, gain)
            print(f"High !!!!")

        if norm_rot < rot_threshold:
            rot_gain = max_gain
        else:
            rot_gain = max_gain * numpy.exp(-12 * (norm_rot - rot_threshold))
            rot_gain = numpy.maximum(base, rot_gain)
        return v * gain


    # def velocity_gain_per_axis(v, base_gains=(1.0, 1.0, 1.0), max_gains=(1.3, 1.3, 1.3), thresholds=(0.01, 0.01, 0.01)):
    #     scaled_v = numpy.zeros_like(v)
    #     for i in range(3):
    #         val = abs(v[i])
    #         if val < thresholds[i]:
    #             gain = max_gains[i]
    #         else:
    #             gain = base_gains[i] + (max_gains[i] - base_gains[i]) * (thresholds[i] / val)
    #             gain = min(gain, max_gains[i])
    #         scaled_v[i] = v[i] * gain
    #     return scaled_v

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

    # compute relative orientation of mtm1 and psm
    def alignment_offset_master1(self):
        _, master1_rotation = self.master1.measured_cp()
        _, puppet_rotation = self.puppet.setpoint_cp()
        alignment_offset = numpy.linalg.inv(master1_rotation) @ puppet_rotation
        return alignment_offset
    
    # compute relative orientation of mtm2 and psm
    def alignment_offset_master2(self):
        _, master2_rotation = self.master2.measured_cp()
        _, puppet_rotation = self.puppet.setpoint_cp()
        alignment_offset = numpy.linalg.inv(master2_rotation) @ puppet_rotation
        return alignment_offset
    
    ############################################################################
    # compute relative orientation of mtm1 and mtm2
    def alignment_offset_master1_to_master2(self):
        _, master1_rotation = self.master1.measured_cp()
        _, master2_rotation = self.master2.measured_cp()
        alignment_offset = numpy.linalg.inv(master1_rotation) @ master2_rotation
        return alignment_offset
    ############################################################################

    # set relative origins for clutching and alignment offset
    def update_initial_state(self):
        # master1
        master1_measured_trans, master1_measured_rot = self.master1.measured_cp()
        self.master1_cartesian_initial = self.set_vctFrm3(rotation=master1_measured_rot, translation=master1_measured_trans)

        # master2
        master2_measured_trans, master2_measured_rot = self.master2.measured_cp()
        self.master2_cartesian_initial = self.set_vctFrm3(rotation=master2_measured_rot, translation=master2_measured_trans)

        # puppet
        puppet_measured_trans, puppet_measured_rot = self.puppet.measured_cp()
        self.puppet_cartesian_initial = self.set_vctFrm3(rotation=puppet_measured_rot, translation=puppet_measured_trans)

        self.alignment_offset_initial_master1 = self.alignment_offset_master1()
        self.alignment_offset_initial_master2 = self.alignment_offset_master2()
        self.alignment_offset_initial_masters = self.alignment_offset_master1_to_master2()    # Used for strong-connected archit.

        self.master1_offset_angle, self.master1_offset_axis = self.GetRotAngle(self.alignment_offset_initial_master1)
        self.master2_offset_angle, self.master2_offset_axis = self.GetRotAngle(self.alignment_offset_initial_master2)
        self.masters_offset_angle, self.masters_offset_axis = self.GetRotAngle(self.alignment_offset_initial_masters)    # Used for strong-connected archit.

    def gripper_to_jaw(self, gripper_angle):
        jaw_angle = self.gripper_to_jaw_scale * gripper_angle + self.gripper_to_jaw_offset
        # make sure we don't set goal past joint limits
        return max(jaw_angle, self.jaw_min)

    def jaw_to_gripper(self, jaw_angle):
        return (jaw_angle - self.gripper_to_jaw_offset) / self.gripper_to_jaw_scale

    # def check_arm_state(self):
    #     if not self.puppet.is_homed():
    #         print(f'ERROR: {self.ral.node_name()}: puppet ({self.puppet.name}) is not homed anymore')
    #         self.running = False
    #     if not self.master1.is_homed():
    #         print(f'ERROR: {self.ral.node_name()}: master ({self.master1.name}) is not homed anymore')
    #         self.running = False
    #     if not self.master2.is_homed():
    #         print(f'ERROR: {self.ral.node_name()}: master ({self.master2.name}) is not homed anymore')
    #         self.running = False

    def enter_aligning(self):
        self.current_state = teleoperation.State.ALIGNING
        self.last_align = None
        self.last_operator_prompt = time.perf_counter()

        # self.master1.arm.use_gravity_compensation(True)
        # self.master2.arm.use_gravity_compensation(True)
        self.puppet.arm.hold()

        # reset operator activity data in case operator is inactive
        self.operator_roll_min_master1 = math.pi * 100
        self.operator_roll_max_master1 = -math.pi * 100
        self.operator_gripper_min_master1 = math.pi * 100
        self.operator_gripper_max_master1 = -math.pi * 100

        self.operator_roll_min_master2 = math.pi * 100
        self.operator_roll_max_master2 = -math.pi * 100
        self.operator_gripper_min_master2 = math.pi * 100
        self.operator_gripper_max_master2 = -math.pi * 100

    def transition_aligning(self):
        if self.operator_is_active and self.on_clutch():
            self.enter_clutched()
            return

        master1_alignment_offset = self.alignment_offset_master1()
        master1_orientation_error, _ = self.GetRotAngle(master1_alignment_offset)
        master2_alignment_offset = self.alignment_offset_master2()
        master2_orientation_error, _ = self.GetRotAngle(master2_alignment_offset)

        master1_aligned = master1_orientation_error <= self.operator_orientation_tolerance
        master2_aligned = master2_orientation_error <= self.operator_orientation_tolerance
        aligned = master1_aligned and master2_aligned
        # print(f"If master1 is aligned: {master1_aligned}")
        # print(f"If master2 is aligned: {master2_aligned}")
        if aligned and self.operator_is_active:
            self.enter_following()

    def run_aligning(self):
        master1_alignment_offset = self.alignment_offset_master1()
        master1_orientation_error, _ = self.GetRotAngle(master1_alignment_offset)
        master2_alignment_offset = self.alignment_offset_master2()
        master2_orientation_error, _ = self.GetRotAngle(master2_alignment_offset)

        # if operator is inactive, use gripper or roll activity to detect when the user is ready
        if self.operator_is_present:
            master1_gripper = self.master1.gripper_measured_js()
            master2_gripper = self.master2.gripper_measured_js()

            self.operator_gripper_max_master1 = max(master1_gripper, self.operator_gripper_max_master1)
            self.operator_gripper_min_master1 = min(master1_gripper, self.operator_gripper_min_master1)
            master1_gripper_range = self.operator_gripper_max_master1 - self.operator_gripper_min_master1

            self.operator_gripper_max_master2 = max(master2_gripper, self.operator_gripper_max_master2)
            self.operator_gripper_min_master2 = min(master2_gripper, self.operator_gripper_min_master2)
            master2_gripper_range = self.operator_gripper_max_master2 - self.operator_gripper_min_master2

            if master1_gripper_range >= self.operator_gripper_threshold or master2_gripper_range >= self.operator_gripper_threshold:
                self.operator_is_active = True

            # determine amount of roll around z axis by rotation of y-axis
            _, master1_rotation = self.master1.measured_cp()
            _, master2_rotation = self.master2.measured_cp()
            _, puppet_rotation = self.puppet.measured_cp()

            master1_y_axis = numpy.array([master1_rotation[0,1], master1_rotation[1,1], master1_rotation[2,1]])
            master2_y_axis = numpy.array([master2_rotation[0,1], master2_rotation[1,1], master2_rotation[2,1]])
            puppet_y_axis = numpy.array([puppet_rotation[0,1], puppet_rotation[1,1], puppet_rotation[2,1]])
            roll_master1 = math.acos(numpy.dot(puppet_y_axis, master1_y_axis))
            roll_master2 = math.acos(numpy.dot(puppet_y_axis, master2_y_axis))

            self.operator_roll_max_master1 = max(roll_master1, self.operator_roll_max_master1)
            self.operator_roll_min_master1 = min(roll_master1, self.operator_roll_min_master1)
            master1_roll_range = self.operator_roll_max_master1 - self.operator_roll_min_master1

            self.operator_roll_max_master2 = max(roll_master2, self.operator_roll_max_master2)
            self.operator_roll_min_master2 = min(roll_master2, self.operator_roll_min_master2)
            master2_roll_range = self.operator_roll_max_master2 - self.operator_roll_min_master2

            if master1_roll_range >= self.operator_roll_threshold or master2_roll_range >= self.operator_roll_threshold:
                self.operator_is_active = True

        # periodically send move_cp to MTM to align with PSM
        master1_aligned = master1_orientation_error <= self.operator_orientation_tolerance
        master2_aligned = master2_orientation_error <= self.operator_orientation_tolerance
        now = time.perf_counter()
        if not self.last_align or now - self.last_align > 4.0:
            _, goal_rotation = self.puppet.setpoint_cp()
            master1_goal_translation, _ = self.master1.setpoint_cp()
            master2_goal_translation, _ = self.master2.setpoint_cp()

            master1_move_goal = self.set_vctFrm3(rotation=goal_rotation, translation=master1_goal_translation)
            master2_move_goal = self.set_vctFrm3(rotation=goal_rotation, translation=master2_goal_translation)

            self.master1.move_cp(master1_move_goal)
            self.master2.move_cp(master2_move_goal)
            self.last_align = now

        # periodically notify operator if un-aligned or operator is inactive
        if self.operator_is_present and now - self.last_operator_prompt > 4.0:
            self.last_operator_prompt = now
            if not master1_aligned:
                print(f'Unable to align master ({self.master1.name}), angle error is {master1_orientation_error * 180 / math.pi} (deg)')
            if not master2_aligned:
                print(f'Unable to align master ({self.master2.name}), angle error is {master2_orientation_error * 180 / math.pi} (deg)')
            elif not self.operator_is_active:
                print(f'To begin teleop, pinch/twist master ({self.master1.name}) or ({self.master2.name}) gripper a bit')

    def enter_clutched(self):
        self.current_state = teleoperation.State.CLUTCHED
        print("enter clutch!")
        # let MTM position move freely, but lock orientation
        wrench = numpy.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.master1.body_servo_cf(wrench)
        self.master2.body_servo_cf(wrench)

        self.master1.lock_orientation()
        self.master2.lock_orientation()
        self.puppet.arm.hold()

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
        jaw_setpoint = self.puppet.jaw_setpoint_js()

        # avoid []
        if len(jaw_setpoint) != 1:
            print(f'Unable to get jaw position. Make sure there is an instrument on the puppet ({self.puppet.name})')
            self.running = False
        self.gripper_ghost = self.jaw_to_gripper(jaw_setpoint)

        # self.master1.arm.use_gravity_compensation(True)
        # self.master2.arm.use_gravity_compensation(True)

    def transition_following(self):
        if not self.on_operator_present():
            self.enter_aligning()
        elif self.on_clutch():
            self.enter_clutched()


    def run_following(self):               
        """
        Forward Process
        """
        # Force channel
        """ recorde cartesian force for plot """
        master1_measured_cf = self.master1.body_measured_cf()   # (6,) numpy array
        master2_measured_cf = self.master2.body_measured_cf()   # (6,) numpy array
        puppet_measured_cf = self.puppet.body_measured_cf()
        """  """



        # master1
        try:
            self.master1_internal_force = master1_internal_f_queue.get_nowait()
            self.master1_internal_force = numpy.asarray(self.master1_internal_force, dtype=float).reshape(-1)
            #print(f"internal{self.master1_internal_force}")
        except Empty:
            pass
        
        # master2
        try:
            self.master2_internal_force = master2_internal_f_queue.get_nowait()
            self.master2_internal_force = numpy.asarray(self.master2_internal_force, dtype=float).reshape(-1)
        except Empty:
            pass

        # puppet
        try:
            self.puppet_internal_force = puppet_internal_f_queue.get_nowait()
            self.puppet_internal_force = numpy.asarray(self.puppet_internal_force, dtype=float).reshape(-1)
        except Empty:
            pass

        # master1
        try:
            master1_external_f = master1_external_f_queue.get_nowait()
            master1_external_f = numpy.asarray(master1_external_f, dtype=float).reshape(-1)
        except Empty:
            master1_external_f = numpy.asarray(master1_measured_cf, dtype=float).reshape(-1).copy()
            master1_external_f -= self.master1_internal_force
            #print(f"external:{self.master1_internal_force}")
            master1_external_f[:3] *= -1.0
            master1_external_f[3:6] = 0.0  # turn off torque
        
        # master2
        try:
            master2_external_f = master2_external_f_queue.get_nowait()
            master2_external_f = numpy.asarray(master2_external_f, dtype=float).reshape(-1)
        except Empty:
            master2_external_f = numpy.asarray(master2_measured_cf, dtype=float).reshape(-1).copy()
            master2_external_f -= self.master2_internal_force
            master2_external_f[:3] *= -1.0
            master2_external_f[3:6] = 0.0  # turn off torque
        
        # puppet
        try:
            puppet_external_f = puppet_external_f_queue.get_nowait()
            puppet_external_f = numpy.asarray(puppet_external_f, dtype=float).reshape(-1)
        except Empty:
            puppet_external_f = numpy.asarray(puppet_measured_cf, dtype=float).reshape(-1).copy()
            puppet_external_f -= self.puppet_internal_force
            puppet_external_f[:3] *= -1.0
            puppet_external_f[3:6] = 0.0  # turn off torque



        
        # force input
        gamma = 1.0
        force_goal = self.force_gain * (self.alpha * master1_external_f + (1 - self.alpha) * master2_external_f + gamma * puppet_external_f)
        force_goal = force_goal.reshape(-1)

        # Position channel
        master1_measured_trans, master1_measured_rot = self.master1.measured_cp()
        master2_measured_trans, master2_measured_rot = self.master2.measured_cp()
        master1_initial_trans = self.master1_cartesian_initial.GetTranslation()   ####### Reference or copy #########
        master2_initial_trans = self.master2_cartesian_initial.GetTranslation()
        puppet_initial_trans = self.puppet_cartesian_initial.GetTranslation()


        # set translation of psm
        master1_translation = master1_measured_trans - master1_initial_trans
        master2_translation = master2_measured_trans - master2_initial_trans
        master1_translation *= self.scale
        master2_translation *= self.scale
        master_total_translation = self.alpha * master1_translation + (1 - self.alpha) * master2_translation
        puppet_position = master_total_translation + puppet_initial_trans

        # set rotation of psm to match mtm plus alignment offset
        # if we can actuate the MTM, we slowly reduce the alignment offset to zero over time
        max_delta = self.align_rate * self.run_period
        self.master1_offset_angle += math.copysign(min(abs(self.master1_offset_angle), max_delta), -self.master1_offset_angle)
        self.master2_offset_angle += math.copysign(min(abs(self.master2_offset_angle), max_delta), -self.master2_offset_angle)
        
        # rotation offset from master1
        master1_alignment_offset = self.GetRotMatrix(self.master1_offset_axis, self.master1_offset_angle)
        master1_rotation_alignment = master1_measured_rot @ master1_alignment_offset

        # rotation offset from master2
        master2_alignment_offset = self.GetRotMatrix(self.master2_offset_axis, self.master2_offset_angle)
        master2_rotation_alignment = master2_measured_rot @ master2_alignment_offset

        # average rotation
        puppet_rotation = self.average_rotation(master1_rotation_alignment, master2_rotation_alignment, alpha=self.alpha)

        # set cartesian goal of psm
        puppet_cartesian_goal = self.set_vctFrm3(rotation=puppet_rotation, translation=puppet_position)


        # Velocity channel
        master1_measured_cv = self.master1.measured_cv()   # (6,) numpy array
        master1_measured_cv[0:3] *= self.velocity_scale      # scale the linear velocity
        master1_measured_cv[3:6] *= 1.0      # scale down the angular velocity by 0.8

        master2_measured_cv = self.master2.measured_cv()   # master2
        master2_measured_cv[0:3] *= self.velocity_scale     
        master2_measured_cv[3:6] *= 1.0     

        raw_puppet_velocity_goal = self.alpha * master1_measured_cv + (1 - self.alpha) * master2_measured_cv
        puppet_velocity_goal = self.set_velocity_goal(v=raw_puppet_velocity_goal)

        # Move
        self.puppet.servo_cs(puppet_cartesian_goal, puppet_velocity_goal, force_goal)


        ### Jaw/gripper teleop
        current_master1_gripper = self.master1.gripper_measured_js()
        current_master2_gripper = self.master2.gripper_measured_js()

        master1_ghost_lag = current_master1_gripper - self.gripper_ghost
        master2_ghost_lag = current_master2_gripper - self.gripper_ghost
        # average gripper lag
        ghost_lag = self.alpha * master1_ghost_lag + (1 - self.alpha) * master2_ghost_lag

        max_delta = self.jaw_rate * self.run_period
        # move ghost at most max_delta towards current gripper
        self.gripper_ghost += math.copysign(min(abs(ghost_lag), max_delta), ghost_lag)
        jaw_goal = numpy.array([self.gripper_to_jaw(self.gripper_ghost)]).reshape(-1)
        self.puppet.jaw_servo_jp(jaw_goal)


        """
        Backward Process
        """
        # Position channel
        puppet_measured_trans, puppet_measured_rot = self.puppet.measured_cp()
        puppet_translation = puppet_measured_trans - puppet_initial_trans     ##### should it update puppet initial cartesian after forward process???
        puppet_translation /= self.scale

        # set translation of mtm1
        master1_position = puppet_translation + master1_initial_trans
        # set translation of mtm2
        master2_position = puppet_translation + master2_initial_trans

        # set rotation of mtm1
        master1_rotation = puppet_measured_rot @ numpy.linalg.inv(master1_alignment_offset)
        # set rotation of mtm2
        master2_rotation = puppet_measured_rot @ numpy.linalg.inv(master2_alignment_offset)

        # set cartesian goal of mtm1 and mtm2
        master1_cartesian_goal = self.set_vctFrm3(rotation=master1_rotation, translation=master1_position)
        master2_cartesian_goal = self.set_vctFrm3(rotation=master2_rotation, translation=master2_position)


        # Velocity channel
        puppet_measured_cv = self.puppet.measured_cv()   # (6,) numpy array
        puppet_measured_cv[0:3] /= self.velocity_scale      # scale the linear velocity
        # puppet_measured_cv[3:6] *= 0.8      # scale down the angular velocity by 0.8

        # set velocity goal
        # master1_velocity_goal = self.velocity_gain * (puppet_measured_cv + master2_measured_cv) / 2.0
        # master2_velocity_goal = self.velocity_gain * (puppet_measured_cv + master1_measured_cv) / 2.0

        master1_velocity_goal = self.set_velocity_goal(v=puppet_measured_cv)
        master2_velocity_goal = self.set_velocity_goal(v=puppet_measured_cv)

        # Move
        self.master1.servo_cs(master1_cartesian_goal, master1_velocity_goal, force_goal)
        self.master2.servo_cs(master2_cartesian_goal, master2_velocity_goal, force_goal)


        """
        record measured cartesian force
        """
        self.total_force_MTML.append(master1_measured_cf.copy())
        self.total_force_MTMR.append(master2_measured_cf.copy())
        self.total_force_PSM.append(puppet_measured_cf.copy())

        """
        record position tracking
        """
        self.y_data_l.append(puppet_measured_trans.copy())
        self.y_data_l_expected.append(puppet_position.copy())



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
        self.puppet.move_jp(puppet_initial_position)
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
            model_thread.stop()
            system.power_off()


        # save data
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_29/multi_array.txt', self.y_data_l, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_29/multi_array_exp.txt', self.y_data_l_expected, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_29/MTML_total_force.txt', self.total_force_MTML, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_29/MTMR_total_force.txt', self.total_force_MTMR, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/July_29/PSM_total_force.txt', self.total_force_PSM, fmt='%f', delimiter=' ', comments='')


        numpy.savetxt('/home/xle6/dvrk_teleop_data/Aug_18/Model_performance/PSM_internal.txt', internal_torque_record_PSM, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/Aug_18/Model_performance/PSM_total.txt', total_torque_record_PSM, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/Aug_18/Model_performance/PSM_force.txt', internal_force_PSM, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/Aug_18/Model_performance/MTML_internal.txt', internal_torque_record_MTML, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/Aug_18/Model_performance/MTML_total.txt', total_torque_record_MTML, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/Aug_18/Model_performance/MTML_force.txt', internal_force_MTML, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/Aug_18/Model_performance/MTMR_internal.txt', internal_torque_record_MTMR, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/Aug_18/Model_performance/MTMR_total.txt', total_torque_record_MTMR, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/Aug_18/Model_performance/MTMR_force.txt', internal_force_MTMR, fmt='%f', delimiter=' ', comments='')

        numpy.savetxt('/home/xle6/dvrk_teleop_data/Aug_18/Model_performance/MTML_CF.txt', cartesian_force_MTML, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/Aug_18/Model_performance/MTMR_CF.txt', cartesian_force_MTMR, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt('/home/xle6/dvrk_teleop_data/Aug_18/Model_performance/PSM_CF.txt', cartesian_force_PSM, fmt='%f', delimiter=' ', comments='')
        print(f"data.txt saved!")

class ARM:
    class LoadModel:
        def __init__(self, onnx_path, param_path):
            self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.onnx_path = onnx_path
            self.param_path = param_path
            norm_data = numpy.load(param_path)
            self.input_mean = norm_data['input_mean']
            self.input_std = norm_data['input_std']
            self.target_mean = norm_data['target_mean']
            self.target_std = norm_data['target_std']
            self.seq_len = norm_data['seq_len']

    def __init__(self, arm, name, firstjoints_onnxpath=None, firstjoints_parampath=None, lastjoints_onnxpath=None, lastjoints_parampath=None):
        self.arm = arm
        self.name = name

        # load onnx model
        if firstjoints_onnxpath is not None and firstjoints_parampath is not None:
            self.firstmodel = self.LoadModel(firstjoints_onnxpath, firstjoints_parampath)

        if lastjoints_onnxpath is not None and lastjoints_parampath is not None:
            self.lastmodel = self.LoadModel(lastjoints_onnxpath, lastjoints_parampath)

        # network parameters setting ######################################################
        self.internal_torque_record_MTML= []
        self.total_torque_record_MTML = []
        self.internal_torque_record_MTMR = []
        self.total_torque_record_MTMR = []
        self.internal_torque_record_PSM = []
        self.total_torque_record_PSM = []

        self.internal_force_MTML = []
        self.internal_force_MTMR = []
        self.internal_force_PSM = []

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
        assert self.name.startswith(("MTM"))
        gripper = self.arm.gripper.measured_js()
        gripper_position = gripper.Position()
        return gripper_position.copy()
    
    def jaw_setpoint_js(self):
        assert self.name in ("PSM1", "PSM2")
        jaw = self.arm.jaw.setpoint_js()
        jaw_setpoint = jaw.Position()
        return jaw_setpoint.copy()
       
    def jaw_servo_jp(self, goal):
        assert self.name in ("PSM1", "PSM2")
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

    def body_jacobian(self):
        j = self.arm.body.jacobian()
        return j.copy()
    
    def move_jp(self, goal):
        arg = self.arm.move_jp.GetArgumentPrototype()
        arg.SetGoal(goal)
        self.arm.move_jp(arg)

class model(threading.Thread):
    def __init__(self, master1, master2, puppet, freq):
        super().__init__()
        self.interval = 1/freq
        self.running = True
        self.master1 = master1
        self.master2 = master2
        self.puppet = puppet
        self.count = 0

        # # network parameters setting ######################################################
        # self.internal_torque_record_MTML= []
        # self.total_torque_record_MTML = []
        # self.internal_torque_record_MTMR = []
        # self.total_torque_record_MTMR = []
        # self.internal_torque_record_PSM = []
        # self.total_torque_record_PSM = []

        # self.internal_force_MTML = []
        # self.internal_force_MTMR = []
        # self.internal_force_PSM = []

        # network parameters setting ######################################################
        self.seq_len = self.master1.firstmodel.seq_len
        self.queue_MTML_first3 = numpy.zeros((1, self.seq_len, 6))
        self.queue_MTML_last3 = numpy.zeros((1, self.seq_len, 6))
        self.queue_MTMR_first3 = numpy.zeros((1, self.seq_len, 6))
        self.queue_MTMR_last3 = numpy.zeros((1, self.seq_len, 6))
        self.queue_PSM_first3 = numpy.zeros((1, self.seq_len, 6))
        self.queue_PSM_last3 = numpy.zeros((1, self.seq_len, 6))
        

    def run(self):
        time.sleep(15)
        while self.running:
            # print(f"model is running at {1/self.interval}hz")
            start_time = time.time()


            master1_external_f, master2_external_f, puppet_external_f, master1_internal_f, master2_internal_f, puppet_internal_f = self.externalforce_prediction()   # (6,) numpy array for each master and puppet

            # master1
            master1_external_f[0:3] *= -1.0
            master1_external_f[3:6] *= 0   # turn off torque

            # master2
            master2_external_f[0:3] *= -1.0
            master2_external_f[3:6] *= 0  # turn off torque

            # puppet
            puppet_external_f[0:3] *= -1.0
            puppet_external_f[3:6] *= 0

             # master1
            if master1_external_f_queue.full():
                _ = master1_external_f_queue.get_nowait()
            master1_external_f_queue.put(master1_external_f)
            
             # master2
            if master2_external_f_queue.full():
                _ = master2_external_f_queue.get_nowait()
            master2_external_f_queue.put(master2_external_f)
            
             # puppet
            if puppet_external_f_queue.full():
                _ = puppet_external_f_queue.get_nowait()
            puppet_external_f_queue.put(puppet_external_f)

            

            # master1
            if master1_internal_f_queue.full():
                _ = master1_internal_f_queue.get_nowait()
            master1_internal_f_queue.put(master1_internal_f)
            
             # master2
            if master2_internal_f_queue.full():
                _ = master2_internal_f_queue.get_nowait()
            master2_internal_f_queue.put(master2_internal_f)
            
             # puppet
            if puppet_internal_f_queue.full():
                _ = puppet_internal_f_queue.get_nowait()
            puppet_internal_f_queue.put(puppet_internal_f)



            elapsed = time.time() - start_time
            if self.count == 200:
                print(f"Model runs {1/elapsed} times per sec.")
                self.count = 0
            self.count += 1
            time.sleep(max(0, self.interval - elapsed))
    
    def externalforce_prediction(self):

        master_1_force = self.master1.body_measured_cf()
        master_2_force = self.master2.body_measured_cf()
        puppet_force = self.puppet.body_measured_cf()
        cartesian_force_MTML.append(master_1_force)
        cartesian_force_MTMR.append(master_2_force)
        cartesian_force_PSM.append(puppet_force)

        # measured_js returns 6 joints for PSM, 7 joints for MTM
        # MTM1
        master1_measured_q, master1_measured_dq, master1_measured_torque = self.master1.measured_js()
        master1_q = master1_measured_q[:6]
        master2_measured_q, master2_measured_dq, master2_measured_torque = self.master2.measured_js()
        master2_q = master2_measured_q[:6]
        puppet_measured_q, puppet_measured_dq, puppet_measured_torque = self.puppet.measured_js()
        puppet_q = puppet_measured_q[:6]

        # if not master1_q.any() or not master2_q.any() or not puppet_q.any():
        #     ans = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        #     return (ans, ans, ans)




        master1_dq = master1_measured_dq[:6]
        master1_total_torque = master1_measured_torque[:6]

        # MTM2
        master2_dq = master2_measured_dq[:6]
        master2_total_torque = master2_measured_torque[:6]

        # PSM
        puppet_dq = puppet_measured_dq[:6]
        puppet_total_torque = puppet_measured_torque[:6]
        # print(f"{component.name} measured_jf is: {total_torque}")

        # Concat
        master1_first_input = numpy.concatenate((master1_q[0:3], master1_dq[0:3]))
        master1_last_input = numpy.concatenate((master1_q[3:6], master1_dq[3:6]))

        master2_first_input = numpy.concatenate((master2_q[0:3], master2_dq[0:3]))
        master2_last_input = numpy.concatenate((master2_q[3:6], master2_dq[3:6]))

        puppet_first_input = numpy.concatenate((puppet_q[0:3], puppet_dq[0:3]))
        puppet_last_input = numpy.concatenate((puppet_q[3:6], puppet_dq[3:6]))

        # print(f"input mean is: {component.firstmodel.input_mean}")
        # print(f"input std is: {component.firstmodel.input_std}")
        # normalize input
        master1_first_input = (master1_first_input - self.master1.firstmodel.input_mean) / self.master1.firstmodel.input_std
        master1_last_input = (master1_last_input - self.master1.lastmodel.input_mean) / self.master1.lastmodel.input_std

        master2_first_input = (master2_first_input - self.master2.firstmodel.input_mean) / self.master2.firstmodel.input_std
        master2_last_input = (master2_last_input - self.master2.lastmodel.input_mean) / self.master2.lastmodel.input_std

        puppet_first_input = (puppet_first_input - self.puppet.firstmodel.input_mean) / self.puppet.firstmodel.input_std
        puppet_last_input = (puppet_last_input - self.puppet.lastmodel.input_mean) / self.puppet.lastmodel.input_std

        # reshape
        master1_first_input = numpy.expand_dims(master1_first_input.reshape(1,-1), axis=0)    # shape(1,1,6)
        master1_last_input = numpy.expand_dims(master1_last_input.reshape(1,-1), axis=0)    # shape(1,1,6)

        master2_first_input = numpy.expand_dims(master2_first_input.reshape(1,-1), axis=0)    # shape(1,1,6)
        master2_last_input = numpy.expand_dims(master2_last_input.reshape(1,-1), axis=0)    # shape(1,1,6)

        puppet_first_input = numpy.expand_dims(puppet_first_input.reshape(1,-1), axis=0)    # shape(1,1,6)
        puppet_last_input = numpy.expand_dims(puppet_last_input.reshape(1,-1), axis=0)    # shape(1,1,6)

        self.queue_MTML_first3 = numpy.concatenate((self.queue_MTML_first3, master1_first_input), axis=1)
        self.queue_MTML_last3 = numpy.concatenate((self.queue_MTML_last3, master1_last_input), axis = 1)
        self.queue_MTML_first3 = self.queue_MTML_first3[:, 1:, :]
        self.queue_MTML_last3 = self.queue_MTML_last3[:, 1:, :]

        master1_first_ort_inputs = self.queue_MTML_first3.astype(numpy.float32)
        master1_last_ort_inputs = self.queue_MTML_last3.astype(numpy.float32)

        self.queue_MTMR_first3 = numpy.concatenate((self.queue_MTMR_first3, master2_first_input), axis=1)
        self.queue_MTMR_last3 = numpy.concatenate((self.queue_MTMR_last3, master2_last_input), axis = 1)
        self.queue_MTMR_first3 = self.queue_MTMR_first3[:, 1:, :]
        self.queue_MTMR_last3 = self.queue_MTMR_last3[:, 1:, :]

        master2_first_ort_inputs = self.queue_MTMR_first3.astype(numpy.float32)
        master2_last_ort_inputs = self.queue_MTMR_last3.astype(numpy.float32)

        self.queue_PSM_first3 = numpy.concatenate((self.queue_PSM_first3, puppet_first_input), axis=1)
        self.queue_PSM_last3 = numpy.concatenate((self.queue_PSM_last3, puppet_last_input), axis = 1)
        self.queue_PSM_first3 = self.queue_PSM_first3[:, 1:, :]
        self.queue_PSM_last3 = self.queue_PSM_last3[:, 1:, :]

        puppet_first_ort_inputs = self.queue_PSM_first3.astype(numpy.float32)
        puppet_last_ort_inputs = self.queue_PSM_last3.astype(numpy.float32)

        input_names = [input.name for input in ort_session.get_inputs()]
        inputs_merge = [master1_first_ort_inputs, master1_last_ort_inputs, master2_first_ort_inputs, 
                        master2_last_ort_inputs, puppet_first_ort_inputs, puppet_last_ort_inputs]
        # print(f"input names: {input_names}")
        inputs = {
            name: inputs_merge[i] for i, name in enumerate(input_names)
        }

        model_name = model_merger.model_names
        output_names = [f"{name}output" for name in model_name]

        # ### output name test
        # for output in ort_session.get_outputs():
        #     print(output.name)


        outputs = ort_session.run(output_names, inputs) # m1_first, m1_last, m2_first, m2_last, p_first, p_last

        master1_first_ort_outs = outputs[0]
        master1_last_ort_outs = outputs[1]
        master2_first_ort_outs = outputs[2]
        master2_last_ort_outs= outputs[3]
        puppet_first_ort_outs = outputs[4]
        puppet_last_ort_outs = outputs[5]

        # denormalize output
        master1_torque_Joint1_3 = master1_first_ort_outs[0]
        master1_torque_Joint4_6 = master1_last_ort_outs[0]

        master2_torque_Joint1_3 = master2_first_ort_outs[0]
        master2_torque_Joint4_6 = master2_last_ort_outs[0]

        puppet_torque_Joint1_3 = puppet_first_ort_outs[0]
        puppet_torque_Joint4_6 = puppet_last_ort_outs[0]

        master1_torque_Joint1_3 = master1_torque_Joint1_3 * self.master1.firstmodel.target_std + self.master1.firstmodel.target_mean
        master1_torque_Joint4_6 = master1_torque_Joint4_6 * self.master1.lastmodel.target_std + self.master1.lastmodel.target_mean

        master2_torque_Joint1_3 = master2_torque_Joint1_3 * self.master2.firstmodel.target_std + self.master2.firstmodel.target_mean
        master2_torque_Joint4_6 = master2_torque_Joint4_6 * self.master2.lastmodel.target_std + self.master2.lastmodel.target_mean

        puppet_torque_Joint1_3 = puppet_torque_Joint1_3 * self.puppet.firstmodel.target_std + self.puppet.firstmodel.target_mean
        puppet_torque_Joint4_6 = puppet_torque_Joint4_6 * self.puppet.lastmodel.target_std + self.puppet.lastmodel.target_mean

        master1_internal_torque = numpy.hstack((master1_torque_Joint1_3, master1_torque_Joint4_6))
        master1_external_torque = (master1_total_torque - master1_internal_torque)

        master2_internal_torque = numpy.hstack((master2_torque_Joint1_3, master2_torque_Joint4_6))
        master2_external_torque = (master2_total_torque - master2_internal_torque)

        puppet_internal_torque = numpy.hstack((puppet_torque_Joint1_3, puppet_torque_Joint4_6))
        puppet_external_torque = (puppet_total_torque - puppet_internal_torque)

        # convert to cartesian force

        # print(f"master1_external_torque : {master1_external_torque}")
        # prinyt(f"measured torque = {master1_measured_torque}")
        master1_external_torque = numpy.append(master1_external_torque,master1_measured_torque[6])
        master1_internal_torque = numpy.append(master1_internal_torque,master1_measured_torque[6])
        master1_J = self.master1.body_jacobian()   # shape (6,7) of MTM and (6,6) of PSM
        master1_external_force = numpy.linalg.pinv(master1_J.T) @ master1_external_torque.T

        master2_external_torque = numpy.append(master2_external_torque,master2_measured_torque[6])
        master2_internal_torque = numpy.append(master2_internal_torque,master2_measured_torque[6])
        master2_J = self.master2.body_jacobian()   # shape (6,7) of MTM and (6,6) of PSM
        master2_external_force = numpy.linalg.pinv(master2_J.T) @ master2_external_torque.T

        puppet_J = self.puppet.body_jacobian()   # shape (6,6) of PSM
        puppet_external_force = numpy.linalg.pinv(puppet_J.T) @ puppet_external_torque.T

        # """ recored inferred force for plot """
        master1_internal_force = numpy.linalg.pinv(master1_J.T) @ master1_internal_torque.T
        master2_internal_force = numpy.linalg.pinv(master2_J.T) @ master2_internal_torque.T
        puppet_internal_force = numpy.linalg.pinv(self.puppet.body_jacobian().T) @ puppet_internal_torque.T
        # print(f"internal force: {internal_force}")
        # print(f"internal force shape: {internal_force.shape}")

        #PSM
        internal_torque_record_PSM.append(puppet_internal_torque.reshape(-1).tolist())
        total_torque_record_PSM.append(puppet_total_torque)
        internal_force_PSM.append(puppet_internal_force.reshape(-1).tolist())

        # MTML
        internal_torque_record_MTML.append(master1_internal_torque.reshape(-1).tolist())
        total_torque_record_MTML.append(master1_total_torque)
        internal_force_MTML.append(master1_internal_force.reshape(-1).tolist())

        #MTMR
        internal_torque_record_MTMR.append(master2_internal_torque.reshape(-1).tolist())
        total_torque_record_MTMR.append(master2_total_torque)
        internal_force_MTMR.append(master2_internal_force.reshape(-1).tolist())

        print(f"master2_internal_torque{master2_internal_torque}")
   
        return (master1_external_force,master2_external_force, puppet_external_force, master1_internal_force, master2_internal_force, puppet_internal_force)   # (6,) numpy array for each master and puppet




    def stop(self):
        self.running = False

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alpha', type = float, required = False, default= 0.5,
                         help = 'dominance factor, between 0 and 1')
    parser.add_argument('-n', '--no-mtm-alignment', action='store_true',
                        help="don't align mtm (useful for using haptic devices as MTM which don't have wrist actuation)")
    parser.add_argument('-i', '--interval', type=float, default=0.00118,
                        help = 'time interval/period to run at - should be as long as system\'s period to prevent timeouts')
    args = parser.parse_args()

    from dvrk_system import *
    path_root = "/home/xle6/dvrk_teleop_data/Aug_18/checkpoints/"

    # mtml1 = ARM(MTML1, 'MTML', 
    #            firstjoints_onnxpath=path_root+"master1_l-First.onnx", 
    #            firstjoints_parampath=path_root+"master1_l-First-stat_params.npz", lastjoints_onnxpath=path_root+"master1_l-Last.onnx", 
    #            lastjoints_parampath=path_root+"master1_l-Last-stat_params.npz")
    # mtml2 = ARM(MTML2, 'MTML-Si',
    #            firstjoints_onnxpath=path_root+"master2_l-First.onnx", 
    #            firstjoints_parampath=path_root+"master2_l-First-stat_params.npz", lastjoints_onnxpath=path_root+"master2_l-Last.onnx", 
    #            lastjoints_parampath=path_root+"master2_l-Last-stat_params.npz")
    # psm2 = ARM(PSM2, 'PSM2',
    #           firstjoints_onnxpath=path_root+"puppet_l-First.onnx", 
    #            firstjoints_parampath=path_root+"puppet_l-First-stat_params.npz", lastjoints_onnxpath=path_root+"puppet_l-Last.onnx", 
    #            lastjoints_parampath=path_root+"puppet_l-Last-stat_params.npz")
    mtml1 = ARM(MTMR1, 'MTMR', 
               firstjoints_onnxpath=path_root+"master1_r-First.onnx", 
               firstjoints_parampath=path_root+"master1_r-First-stat_params.npz", lastjoints_onnxpath=path_root+"master1_r-Last.onnx", 
               lastjoints_parampath=path_root+"master1_r-Last-stat_params.npz")
    mtml2 = ARM(MTMR2, 'MTMR-Si',
               firstjoints_onnxpath=path_root+"master2_r-First.onnx", 
               firstjoints_parampath=path_root+"master2_r-First-stat_params.npz", lastjoints_onnxpath=path_root+"master2_r-Last.onnx", 
               lastjoints_parampath=path_root+"master2_r-Last-stat_params.npz")
    psm2 = ARM(PSM1, 'PSM1',
              firstjoints_onnxpath=path_root+"puppet_r-First.onnx", 
               firstjoints_parampath=path_root+"puppet_r-First-stat_params.npz", lastjoints_onnxpath=path_root+"puppet_r-Last.onnx", 
               lastjoints_parampath=path_root+"puppet_r-Last-stat_params.npz")


    clutch = clutch
    coag = coag


    application = teleoperation(mtml1, mtml2, psm2, clutch, args.interval,
                                not args.no_mtm_alignment, operator_present_topic = coag, alpha = args.alpha)
    
    # right hand
    # application = teleoperation(mtmr2, mtmr1, psm1, clutch, args.interval,
    #                             not args.no_mtm_alignment, operator_present_topic = coag, alpha = args.alpha)
    
    # merge models
    merged_name = "mergerd_model.onnx"
    model_paths = (mtml1.firstmodel.onnx_path,mtml1.lastmodel.onnx_path, mtml2.firstmodel.onnx_path ,mtml2.lastmodel.onnx_path,
                       psm2.firstmodel.onnx_path, psm2.lastmodel.onnx_path)
    # # right model
    # model_paths = (mtmr2.firstmodel.onnx_path,mtmr2.lastmodel.onnx_path, mtmr1.firstmodel.onnx_path ,mtmr1.lastmodel.onnx_path,
    #                    psm1.firstmodel.onnx_path, psm1.lastmodel.onnx_path)
    model_merger = ModelMerger(model_paths,path_root + merged_name)
    print("Merging models...")
    model_merger.merge_models()
    print("Models merged successfully.")

    # load merged model
    ort_session = onnxruntime.InferenceSession(path_root + merged_name, providers=["CPUExecutionProvider"])
    model_thread = model(mtml1, mtml2, psm2, freq=200)  
    model_thread.start()
    application.run()
    
     
    system.power_off()
