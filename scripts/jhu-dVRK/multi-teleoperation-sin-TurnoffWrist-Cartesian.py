#!/usr/bin/env python

# Author: Junxiang Wang
# Date: 2024-04-12

# (C) Copyright 2024-2025 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

"""For instructions, see https://dvrk.readthedocs.io, and search for \"dvrk_teleoperation\""""
"""correct the control law, and turn off force feedback on wrist axes of two MTMs (Optional)"""

import argparse
# import crtk
from enum import Enum
import math
# import std_msgs.msg
import sys
import time
from dvrk_console import *
#from dvrk_console_multi import *
import cisstVectorPython as cisstVector
import pdb
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.spatial.transform.rotation import Rotation


'''Low Pass Filter'''
class LowPassFilter:
    def __init__(self, alpha: float, dim:int, init_val=None):
        self.alpha = alpha
        self.y_prev = numpy.zeros(dim) if init_val is None else numpy.array(init_val, dtype = float)

    def update(self, x_curr):
        x_curr = numpy.array(x_curr, dtype = float)
        self.y_prev = self.alpha * x_curr + (1.0 - self.alpha) * self.y_prev
        return self.y_prev
    



class teleoperation:
    class State(Enum):
        ALIGNING = 1
        CLUTCHED = 2
        FOLLOWING = 3

    def __init__(self, mtml1, mtml2, psm2, clutch, coag, alpha, beta, clutch_topic, run_period, align_mtm, operator_present_topic = ""):
        self.force_filter = LowPassFilter(alpha=0.1, dim=6)
        self.velocity_filter = LowPassFilter(alpha=0.2, dim=3)
        self.puppet_velocity_filter = LowPassFilter(alpha=0.2, dim=3)
        # print('Initialzing dvrk_teleoperation for {} and {}'.format(master1.name, puppet.name))
        print(f"running at {1/run_period} frequency")
        self.run_period = run_period

        # dominance factor
        self.alpha = alpha
        self.beta = beta

        # MTML - PSM2
        self.master1 = mtml1 # MTML1
        self.master2 = mtml2 # MTML2
        self.puppet = psm2 # PSM2 - MTML 

        self.clutch = clutch
        self.coag = coag

        # operating state
        self.master1_op_state = self.master1.operating_state()
        self.master1_is_busy = True

        self.scale = 0.2

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

        self.operator_is_active = True
        if operator_present_topic:
            self.operator_is_present = False

        self.clutch_pressed = False
        self.a = 0

        # for plotting
        self.time_data = []
        self.y_data_l = []
        self.y_data_l_expected = []
        self.m1_force = []
        self.m2_force = []
        self.puppet_force = []


    def update(self, frames, input, time_data, y_data, line):
        current_time = time.time()  # Get the current timestamp
        y_value = input  # Get the real-time data (replace this with your actual data)

        # Append the new time and data to their respective lists
        time_data.append(current_time)
        y_data.append(y_value)

        # Update the data of the line on the plot
        line.set_data(time_data, y_data)

        # Adjust x-axis limits to show the latest data
        if current_time >= self.ax.get_xlim()[1]:
            self.ax.set_xlim(time_data[0], current_time + 10)

        return line,
        
    def GetRotAngle(self, R):
        
        # Extract the rotation angle (theta)
        theta = numpy.arccos((numpy.trace(R) - 1) / 2)
        
        # Handle edge case when the angle is 0 or 180 degrees
        if numpy.isclose(theta, 0):
            # Identity matrix, no rotation
            axis = numpy.array([1, 0, 0])  # Arbitrary
        elif numpy.isclose(theta, numpy.pi):
            # 180-degree rotation, axis is arbitrary
            axis = numpy.sqrt(numpy.diagonal(R) / 2)
        else:
            # Extract the rotation axis
            axis = numpy.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ]) / (2 * numpy.sin(theta))
        
        # Normalize the axis
        axis = axis / numpy.linalg.norm(axis)
        
        return theta, axis


    def GetRotMatrix(self, axis, theta):
        # Ensure the axis is a unit vector
        axis = axis / numpy.linalg.norm(axis)
        
        # Skew-symmetric matrix K from the axis
        K = numpy.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Identity matrix I
        I = numpy.eye(3)
        
        # Rodrigues' rotation formula
        R = I + numpy.sin(theta) * K + (1 - numpy.cos(theta)) * numpy.dot(K, K)
        
        return R

    # average rotation with quaternion
    def average_rotation(self, rotations, alpha):
        # transfrom into quaternion
        quaternions = numpy.array([Rotation.from_dcm(R_i).as_quat() for R_i in rotations])

        # average and norm
        # mean_quat = alpha*(quaternions[0,:]) + (1-alpha)*(quaternions[1,:])
        mean_quat = (quaternions[0,:] + quaternions[1,:]) / 2.0
        mean_quat /= numpy.linalg.norm(mean_quat)

        # transform into rotation matrix
        return Rotation.from_quat(mean_quat).as_dcm()


    # callback for operator pedal/button
    def on_operator_present(self, present):
        self.operator_is_present = present
        if not present:
            self.operator_is_active = False

    # callback for clutch pedal/button
    def on_clutch(self, clutch_pressed):
        self.clutch_pressed = clutch_pressed

    # compute relative orientation of mtm and psm
    def alignment_offset1(self):
        # master
        master_measured_cp = self.master1.measured_cp()
        master_measured_cp_pos = master_measured_cp.Position()
        master_measured_cp_rot = master_measured_cp_pos.GetRotation()
        # puppet
        puppet_measured_cp = self.puppet.setpoint_cp()
        puppet_measured_cp_pos = puppet_measured_cp.Position()
        puppet_measured_cp_rot = puppet_measured_cp_pos.GetRotation()
        return numpy.linalg.inv(master_measured_cp_rot) @ puppet_measured_cp_rot
    
    def alignment_offset2(self):
        # master
        master_measured_cp = self.master2.measured_cp()
        master_measured_cp_pos = master_measured_cp.Position()
        master_measured_cp_rot = master_measured_cp_pos.GetRotation()
        # puppet
        puppet_measured_cp = self.puppet.setpoint_cp()
        puppet_measured_cp_pos = puppet_measured_cp.Position()
        puppet_measured_cp_rot = puppet_measured_cp_pos.GetRotation()
        return numpy.linalg.inv(master_measured_cp_rot) @ puppet_measured_cp_rot
    
    # set relative origins for clutching and alignment offset
    def update_initial_state(self):
        # master1
        self.master_1_cartesian_initial = cisstVector.vctFrm3()
        # measure cp
        m1_measured_cp = self.master1.measured_cp()
        m1_measured_cp_pos = m1_measured_cp.Position()
        m1_measured_cp_rot = m1_measured_cp_pos.GetRotation()
        m1_measured_cp_trans = m1_measured_cp_pos.GetTranslation()
        # set
        self.master_1_cartesian_initial.SetRotation(m1_measured_cp_rot)
        self.master_1_cartesian_initial.SetTranslation(m1_measured_cp_trans)
        #print(f"set to {self.master1.measured_cp().Position().GetTranslation()}")
        #print(f"and is {self.master_1_cartesian_initial.GetTranslation()}")

        # master2
        self.master_2_cartesian_initial = cisstVector.vctFrm3()
        # measure cp
        m2_measured_cp = self.master2.measured_cp()
        m2_measured_cp_pos = m2_measured_cp.Position()
        m2_measured_cp_rot = m2_measured_cp_pos.GetRotation()
        m2_measured_cp_trans = m2_measured_cp_pos.GetTranslation()
        # set
        self.master_2_cartesian_initial.SetRotation(m2_measured_cp_rot)
        self.master_2_cartesian_initial.SetTranslation(m2_measured_cp_trans)

        # puppet
        self.puppet_cartesian_initial = cisstVector.vctFrm3()
        # measure cp
        puppet_measured_cp = self.puppet.setpoint_cp()
        puppet_measured_cp_pos = puppet_measured_cp.Position()
        puppet_measured_cp_rot = puppet_measured_cp_pos.GetRotation()
        puppet_measured_cp_trans = puppet_measured_cp_pos.GetTranslation()
        # set
        self.puppet_cartesian_initial.SetRotation(puppet_measured_cp_rot)
        self.puppet_cartesian_initial.SetTranslation(puppet_measured_cp_trans)

        self.master_1_alignment_offset_initial = self.alignment_offset1()
        self.master_2_alignment_offset_initial = self.alignment_offset2()
        self.master_1_offset_angle, self.master_1_offset_axis = self.GetRotAngle(self.master_1_alignment_offset_initial)
        self.master_2_offset_angle, self.master_2_offset_axis = self.GetRotAngle(self.master_2_alignment_offset_initial)

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
    #         print(f'ERROR: {self.ral.node_name()}: master1 ({self.master1.name}) is not homed anymore')
    #         self.running = False

    def enter_aligning(self):
        self.current_state = teleoperation.State.ALIGNING
        self.last_align = None
        self.last_operator_prompt = time.perf_counter()

        # self.master1.use_gravity_compensation(True)
        # self.master2.use_gravity_compensation(True)
        self.puppet.hold()

        # reset operator activity data in case operator is inactive
        self.operator_roll_min = math.pi * 100
        self.operator_roll_max = -math.pi * 100
        self.operator_gripper_min = math.pi * 100
        self.operator_gripper_max = -math.pi * 100

    def transition_aligning(self):
        # without clutch for debug
        if self.operator_is_active and self.clutch.GetButton():
            self.enter_clutched()
            return

        master_1_alignment_offset = self.alignment_offset1()
        master_2_alignment_offset = self.alignment_offset2()
        master_1_orientation_error, _ = self.GetRotAngle(master_1_alignment_offset)
        master_2_orientation_error, _ = self.GetRotAngle(master_2_alignment_offset)
        aligned = master_1_orientation_error <= self.operator_orientation_tolerance and master_2_orientation_error <= self.operator_orientation_tolerance
        if aligned and self.operator_is_active:
            self.enter_following()

    def run_aligning(self):
        master_1_orientation_error, _ = self.GetRotAngle(self.alignment_offset1())
        master_2_orientation_error, _ = self.GetRotAngle(self.alignment_offset2())

        # if operator is inactive, use gripper or roll activity to detect when the user is ready
        # only detect master1
        if self.coag.GetButton():
            gripper_init = self.master1.gripper.measured_js()
            gripper = gripper_init.Position()
            self.operator_gripper_max = max(gripper, self.operator_gripper_max)
            self.operator_gripper_min = min(gripper, self.operator_gripper_min)
            gripper_range = self.operator_gripper_max - self.operator_gripper_min
            if gripper_range >= self.operator_gripper_threshold:
                self.operator_is_active = True

            # determine amount of roll around z axis by rotation of y-axis
            master_rotation, puppet_rotation = self.master1.measured_cp().Position().GetRotation(), self.puppet.setpoint_cp().Position().GetRotation()
            master_y_axis = numpy.array([master_rotation[0,1], master_rotation[1,1], master_rotation[2,1]])
            puppet_y_axis = numpy.array([puppet_rotation[0,1], puppet_rotation[1,1], puppet_rotation[2,1]])
            roll = math.acos(numpy.dot(puppet_y_axis, master_y_axis))

            self.operator_roll_max = max(roll, self.operator_roll_max)
            self.operator_roll_min = min(roll, self.operator_roll_min)
            roll_range = self.operator_roll_max - self.operator_roll_min
            if roll_range >= self.operator_roll_threshold:
                self.operator_is_active = True

        # periodically send move_cp to MTM to align with PSM
        aligned1 = master_1_orientation_error <= self.operator_orientation_tolerance
        aligned2 = master_2_orientation_error <= self.operator_orientation_tolerance
        now = time.perf_counter()
        # move master1 and master2 spontaneously
        if not self.last_align or now - self.last_align > 4.0:
            # master 1
            move_cp_1 = cisstVector.vctFrm3()
            # setpoint cp
            puppet_setpoint_cp = self.puppet.setpoint_cp()
            m1_setpoint_cp = self.master1.setpoint_cp()
            m2_setpoint_cp = self.master2.setpoint_cp()

            # pos
            puppet_setpoint_pos = puppet_setpoint_cp.Position()
            m1_setpoint_pos = m1_setpoint_cp.Position()
            m2_setpoint_pos = m2_setpoint_cp.Position()

            # rot 
            puppet_setpoint_rot = puppet_setpoint_pos.GetRotation()

            # trans
            m1_setpoint_trans = m1_setpoint_pos.GetTranslation()
            m2_setpoint_trans = m2_setpoint_pos.GetTranslation()

            # set master1
            move_cp_1.SetRotation(puppet_setpoint_rot)
            move_cp_1.SetTranslation(m1_setpoint_trans)
            arg1 = self.master1.move_cp.GetArgumentPrototype()
            arg1.SetGoal(move_cp_1)

            # set master 2
            move_cp_2 = cisstVector.vctFrm3()
            move_cp_2.SetRotation(puppet_setpoint_rot)
            move_cp_2.SetTranslation(m2_setpoint_trans)
            arg2 = self.master2.move_cp.GetArgumentPrototype()
            arg2.SetGoal(move_cp_2)

            self.master1.move_cp(arg1)
            self.master2.move_cp(arg2)
            self.last_align = now

        # periodically notify operator if un-aligned or operator is inactive
        if self.coag.GetButton() and now - self.last_operator_prompt > 4.0:
            self.last_operator_prompt = now
            if not aligned1:
                print(f'Unable to align master1, angle error is {master_1_orientation_error * 180 / math.pi} (deg)')
            elif not aligned2:
                print(f'Unable to align master2, angle error is {master_2_orientation_error * 180 / math.pi} (deg)')
            elif not self.operator_is_active:
                print(f'To begin teleop, pinch/twist master1 gripper a bit')

    def enter_clutched(self):
        self.current_state = teleoperation.State.CLUTCHED

        # let MTM position move freely, but lock orientation
        wrench = numpy.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        arg1 = self.master1.body.servo_cf.GetArgumentPrototype()
        arg1.SetForce(wrench)
        self.master1.body.servo_cf(arg1)
        arg2 = self.master2.body.servo_cf.GetArgumentPrototype()
        arg2.SetForce(wrench)
        self.master2.body.servo_cf(arg2)

        # master1
        m1_lock_cp = self.master1.measured_cp()
        m1_lock_pos = m1_lock_cp.Position()
        m1_lock_rot = m1_lock_pos.GetRotation()
        self.master1.lock_orientation(m1_lock_rot)

        # master2
        m2_lock_cp = self.master2.measured_cp()
        m2_lock_pos = m2_lock_cp.Position()
        m2_lock_rot = m2_lock_pos.GetRotation()
        self.master2.lock_orientation(m2_lock_rot)

        self.puppet.hold()

    def transition_clutched(self):
        if not self.clutch.GetButton() or not self.coag.GetButton():
            self.enter_aligning()

    def run_clutched(self):
        pass

    def enter_following(self):
        self.current_state = teleoperation.State.FOLLOWING
        # update MTM/PSM origins position
        self.update_initial_state()

        # set up gripper ghost to rate-limit jaw speed
        jaw_setpoint = cisstVector.vctFrm3()
        jaw_setpoint_position = self.puppet.jaw.setpoint_js()
        jaw_setpoint = jaw_setpoint_position.Position()
        # prevent []
        if len(jaw_setpoint) == 0:
            jaw_setpoint = numpy.array([0.])
        print(f'jaw_setpoint :{jaw_setpoint}')

        
        # if len(jaw_setpoint) != 1:
        #     print(f'{self.ral.node_name()}: unable to get jaw position. Make sure there is an instrument on the puppet ({self.puppet.name})')
        #     self.running = False
        self.gripper_ghost = self.jaw_to_gripper(jaw_setpoint[0])# convert 1-D array to scalar

        # self.master1.use_gravity_compensation(True)
        # self.master2.use_gravity_compensation(True)

    def transition_following(self):
        if not self.coag.GetButton():
            self.enter_aligning()
        elif self.clutch.GetButton():
            self.enter_clutched()

    def run_following(self):
        ### Cartesian pose teleop
        '''
        Forward Process
        '''
        # Force measurement
        # # master1
        m1_measured_cf = self.master1.body.measured_cf()
        m1_measured_cf_force = m1_measured_cf.Force()
        m1_measured_cf_force[0:3] = m1_measured_cf_force[0:3] * (-1.0)
        m1_measured_cf_force[3:6] = m1_measured_cf_force[3:6] * 0 * 2

        # master1
        '''Measure force from joint space'''
        # m1_measured_js = self.master1.measured_js()
        # m1_measured_jf = m1_measured_js.Effort()
        # m1_measured_jf[-4:] = 0   # turn off force from the last three axis
        # m1_body_jacobian = self.master1.body.jacobian()
        # m1_body_jacobian_trans_inv = numpy.linalg.pinv(m1_body_jacobian.T)
        # m1_measured_cf_force = m1_body_jacobian_trans_inv @ m1_measured_jf
        # m1_measured_cf_force[0:3] = m1_measured_cf_force[0:3] * (-1.0)
        # m1_measured_cf_force[3:6] = m1_measured_cf_force[3:6] * 0 * 2
        # m1_measured_cf_force = self.force_filter.update(m1_measured_cf_force)
        # # master2
        m2_measured_cf = self.master2.body.measured_cf()
        m2_measured_cf_force = m2_measured_cf.Force()
        m2_measured_cf_force[0:3] = m2_measured_cf_force[0:3] * (-1.0)
        m2_measured_cf_force[3:6] = m2_measured_cf_force[3:6] * 0 * 2

        # master2
        '''Measure force from joint space'''
        # m2_measured_js = self.master2.measured_js()
        # m2_measured_jf = m2_measured_js.Effort()
        # m2_measured_jf[-4:] = 0   # turn off force from the last three axis
        # m2_body_jacobian = self.master2.body.jacobian()
        # m2_body_jacobian_trans_inv = numpy.linalg.pinv(m2_body_jacobian.T)
        # m2_measured_cf_force = m2_body_jacobian_trans_inv @ m2_measured_jf
        # m2_measured_cf_force[0:3] = m2_measured_cf_force[0:3] * (-1.0)
        # m2_measured_cf_force[3:6] = m2_measured_cf_force[3:6] * 0 * 2
        # m2_measured_cf_force = self.force_filter.update(m2_measured_cf_force)
        # # puppet1
        puppet_measured_cf = self.puppet.body.measured_cf()
        puppet_measured_cf_force = puppet_measured_cf.Force()
        puppet_measured_cf_force[0:3] = puppet_measured_cf_force[0:3] * 1.0
        puppet_measured_cf_force[3:6] = puppet_measured_cf_force[3:6] * 0 * 2

        # puppet1
        '''Measure force from joint space'''
        # puppet_measured_js = self.puppet.measured_js()
        # puppet_measured_jf = puppet_measured_js.Effort()
        # puppet_measured_jf[-3:] = 0   # turn off force from the last four axis
        # puppet_body_jacobian = self.puppet.body.jacobian()
        # puppet_body_jacobian_trans_inv = numpy.linalg.pinv(puppet_body_jacobian.T)
        # puppet_measured_cf_force = puppet_body_jacobian_trans_inv @ puppet_measured_jf
        # puppet_measured_cf_force[0:3] = puppet_measured_cf_force[0:3] * (-1.0)
        # puppet_measured_cf_force[3:6] = puppet_measured_cf_force[3:6] * 0 * 2
        # puppet_measured_cf_force = self.force_filter.update(puppet_measured_cf_force)
        # force input of the control law
        beta = self.beta
        force_input = 0.2* ( beta * m1_measured_cf_force + (1-beta) * m2_measured_cf_force + puppet_measured_cf_force )


        # Position measurement
        alpha = self.alpha    # position channel dominance factor
        m1_measured_cp = self.master1.measured_cp()
        m2_measured_cp = self.master2.measured_cp()
        master_1_position = m1_measured_cp.Position()
        master_2_position = m2_measured_cp.Position()

        # rot+trans master1
        master_1_rotation1 = master_1_position.GetRotation()
        master_1_trans1 = master_1_position.GetTranslation()
        master_1_trans2 = self.master_1_cartesian_initial.GetTranslation()

        # rot+trans master2
        master_2_rotation1 = master_2_position.GetRotation()
        print(f"master_2_rotation : {master_2_rotation1}")
        master_2_trans1 = master_2_position.GetTranslation()
        master_2_trans2 = self.master_2_cartesian_initial.GetTranslation()

        master_1_translation = master_1_trans1 - master_1_trans2   # relative translation of master1
        master_1_puppet_translation = master_1_translation * self.scale   # convert to puppet frame
        puppet_trans2 = self.puppet_cartesian_initial.GetTranslation()
        master_1_puppet_translation = master_1_puppet_translation + puppet_trans2   # translation input of master1 to puppet

        # translation master2
        master_2_translation = master_2_trans1 - master_2_trans2   # relative translation of master2
        master_2_puppet_translation = master_2_translation * self.scale   # convert to puppet frame
        master_2_puppet_translation += puppet_trans2   # translation input of master2 to puppet

        # average translation (not apply dominance factor alpha for short)
        # puppet_translation = alpha * master_1_puppet_translation + (1-alpha) * master_2_puppet_translation
        puppet_translation = (master_1_puppet_translation + master_2_puppet_translation) / 2.0

        # set rotation of psm to match mtm plus alignment offset
        # if we can actuate the MTM, we slowly reduce the alignment offset to zero over time
        max_delta = self.align_rate * self.run_period
        self.master_1_offset_angle += math.copysign(min(abs(self.master_1_offset_angle), max_delta), -self.master_1_offset_angle)
        self.master_2_offset_angle += math.copysign(min(abs(self.master_2_offset_angle), max_delta), -self.master_2_offset_angle)

        # rotation offset master1
        master_1_alignment_offset = self.GetRotMatrix(self.master_1_offset_axis, self.master_1_offset_angle)
        master_1_puppet_rotation = master_1_rotation1 @ master_1_alignment_offset

        # rotation offset master2
        master_2_alignment_offset = self.GetRotMatrix(self.master_2_offset_axis,self.master_2_offset_angle)
        master_2_puppet_rotation = master_2_rotation1 @ master_2_alignment_offset

        # # average rotation
        puppet_rotation = self.average_rotation(numpy.array([master_1_puppet_rotation,master_2_puppet_rotation]), alpha)
        print(f"puppet_rotation : {puppet_rotation}")



        ##################### use PSM rotation to control PSM
        # puppet_measured_cp_fw = self.puppet.measured_cp()
        # puppet_measured_pos_fw = puppet_measured_cp_fw.Position()
        # puppet_measured_rot_fw = puppet_measured_pos_fw.GetRotation()

        # # average rotation
        # # puppet_rotation = puppet_measured_rot_fw
        # puppet_rotation = self.puppet_measured_rot_fixed
        # print(f"puppet_rotation : {puppet_rotation}")
        #####################



        puppet_cartesian_goal = cisstVector.vctFrm3()
        puppet_cartesian_goal.SetRotation(puppet_rotation)
        puppet_cartesian_goal.SetTranslation(puppet_translation)
        # print(f'puppet_cartesian_goal : {puppet_cartesian_goal}')

        
        # Velocity measurement
        # velocity from master1
        master_1_measured_cv = self.master1.measured_cv()
        master_1_linear_vel = master_1_measured_cv.VelocityLinear()
        master_1_linear_vel = self.velocity_filter.update(master_1_linear_vel)
        master_1_linear_vel = self.scale * master_1_linear_vel
        master_1_angular_vel = master_1_measured_cv.VelocityAngular()

        # velocity from master2
        master_2_measured_cv = self.master2.measured_cv()
        master_2_linear_vel = master_2_measured_cv.VelocityLinear()
        master_2_linear_vel = self.velocity_filter.update(master_2_linear_vel)
        master_2_linear_vel = self.scale * master_2_linear_vel
        master_2_angular_vel = master_2_measured_cv.VelocityAngular()

        # average velocity
        linear_vel_fw = (master_1_linear_vel + master_2_linear_vel) / 2.0
        angular_vel_fw = (master_1_angular_vel + master_2_angular_vel) / 2.0 * 0.0   # zero angular velocity
        vel_fw = numpy.hstack((linear_vel_fw, angular_vel_fw))


        # Force measurement
        average_master_force = force_input


        # execute
        arg_fw = self.puppet.servo_cs.GetArgumentPrototype()
        arg_fw.SetPositionIsValid(True)
        arg_fw.SetPosition(puppet_cartesian_goal)
        #print(f'master_cartesian_goal: {master_cartesian_goal}')
        arg_fw.SetVelocityIsValid(True)
        arg_fw.SetVelocity(vel_fw)
        #print(f'vel_cs : {vel_cs}')
        arg_fw.SetForceIsValid(True)
        arg_fw.SetForce(average_master_force)
        #print(f'force_PSM_cs : {force_PSM_cs}')

        self.puppet.servo_cs(arg_fw)

        # measure puppet force for plot
        puppet_measured_cf_plot = self.puppet.body.measured_cf()
        puppet_measured_force_plot = puppet_measured_cf_plot.Force()
        puppet_measured_force_plot_cat = puppet_measured_force_plot[0:3] * 1
        print(f"arg_fw : {arg_fw}")


        ### Jaw/gripper teleop --- so far only master1 can control the jaw
        # master 1
        master_1_gripper_measured_js_init = self.master1.gripper.measured_js()
        master_1_current_gripper = master_1_gripper_measured_js_init.Position()
        master_1_ghost_lag = master_1_current_gripper - self.gripper_ghost

        # average
        # average_ghost_lag = (master_1_ghost_lag + master_2_ghost_lag) /2.0
        average_ghost_lag = master_1_ghost_lag

        max_delta = self.jaw_rate * self.run_period
        # move ghost at most max_delta towards current gripper
        self.gripper_ghost += math.copysign(min(abs(average_ghost_lag), max_delta), average_ghost_lag)
        
        # gripper_to_jaw = self.gripper_to_jaw(self.gripper_ghost)
        arg = self.puppet.jaw.servo_jp.GetArgumentPrototype()
        arg.SetGoal(numpy.array([self.gripper_to_jaw(self.gripper_ghost)]))
        self.puppet.jaw.servo_jp(arg)
        #print('self.puppet.servo_jp(arg)')


        '''
        Backward Process
        '''
        # Position measurement (only measure puppet's)
        puppet_measured_cp = self.puppet.measured_cp()
        puppet_measured_pos = puppet_measured_cp.Position()
        puppet_measured_rot = puppet_measured_pos.GetRotation()
        puppet_measured_trans = puppet_measured_pos.GetTranslation()

        puppet_relative_translation = puppet_measured_trans - self.puppet_cartesian_initial.GetTranslation()
        master_relative_translation = puppet_relative_translation / self.scale

        # relative trans
        m1_translation_cs = master_relative_translation + self.master_1_cartesian_initial.GetTranslation()
        m2_translation_cs = master_relative_translation + self.master_2_cartesian_initial.GetTranslation()

        # # # relative rot
        m1_rotation_cs = puppet_measured_rot @ numpy.linalg.inv(master_1_alignment_offset)
        m2_rotation_cs = puppet_measured_rot @ numpy.linalg.inv(master_2_alignment_offset)



        # ##################### use MTM1/MTM2 rotation to control MTM1/MTM2
        # m1_measured_cp_bw = self.master1.measured_cp()
        # m2_measured_cp_bw = self.master2.measured_cp()
        # master_1_position_bw = m1_measured_cp_bw.Position()
        # master_2_position_bw = m2_measured_cp_bw.Position()

        # master_1_rotation_bw = master_1_position_bw.GetRotation()
        # master_2_rotation_bw = master_2_position_bw.GetRotation()

        # m1_rotation_cs = master_1_rotation_bw
        # m2_rotation_cs = master_2_rotation_bw
        
        # m1_rotation_cs = self.master_1_rotation1_fixed 
        # m2_rotation_cs = self.master_2_rotation1_fixed
        # # #####################



        # set
        # master1
        m1_cartesian_goal = cisstVector.vctFrm3()
        m1_cartesian_goal.SetRotation(m1_rotation_cs)
        m1_cartesian_goal.SetTranslation(m1_translation_cs)

        # master2
        m2_cartesian_goal = cisstVector.vctFrm3()
        m2_cartesian_goal.SetRotation(m2_rotation_cs)
        m2_cartesian_goal.SetTranslation(m2_translation_cs)


        # Velocity measurement (only measure puppet's)
        puppet_measured_cv = self.puppet.measured_cv()
        linear_vel_cs = puppet_measured_cv.VelocityLinear()
        linear_vel_cs = self.puppet_velocity_filter.update(linear_vel_cs)
        linear_vel_cs = (1/self.scale) * linear_vel_cs * 1.0
        angular_vel_cs = puppet_measured_cv.VelocityAngular()

        angular_vel_cs = angular_vel_cs  * 0.0   # zero angular velocity
        
        vel_cs = numpy.hstack((linear_vel_cs, angular_vel_cs))


        # Force measurement (the same as forward process)
        puppet_measured_cf_force_1 = force_input
        puppet_measured_cf_force_2 = force_input

        # master1 arg
        arg = self.master1.servo_cs.GetArgumentPrototype()
        arg.SetPositionIsValid(True)
        arg.SetPosition(m1_cartesian_goal)
        # print(f'master_cartesian_goal: {m1_cartesian_goal}')
        arg.SetVelocityIsValid(True)
        arg.SetVelocity(vel_cs)
        # print(f'vel_cs : {vel_cs}')
        arg.SetForceIsValid(True)
        arg.SetForce(puppet_measured_cf_force_1)
        self.master1.servo_cs(arg)

        # measure master1 force for plot
        m1_measured_cf_plot = self.master1.body.measured_cf()
        m1_measured_force_plot = m1_measured_cf_plot.Force()
        m1_measured_force_plot = m1_measured_force_plot[0:3] * (-1)

        # master2 arg
        arg2 = self.master2.servo_cs.GetArgumentPrototype()
        arg2.SetPositionIsValid(True)
        arg2.SetPosition(m2_cartesian_goal)
        arg2.SetVelocityIsValid(True)
        arg2.SetVelocity(vel_cs)
        arg2.SetForceIsValid(True)
        arg2.SetForce(puppet_measured_cf_force_2)
        self.master2.servo_cs(arg2)

        # measure master2 force for plot
        m2_measured_cf_plot = self.master2.body.measured_cf()
        m2_measured_force_plot = m2_measured_cf_plot.Force()
        m2_measured_force_plot = m2_measured_force_plot[0:3] * (-1)


        
        # '''
        # plot
        # '''
        puppet_measured_cp_plot = self.puppet.measured_cp()
        puppet_measured_pos_plot = puppet_measured_cp_plot.Position()
        puppet_measured_trans_plot = puppet_measured_pos_plot.GetTranslation()

        self.y_data_l.append(puppet_measured_trans_plot)
        self.y_data_l_expected.append(puppet_translation)

        self.m1_force.append(m1_measured_force_plot)
        self.m2_force.append(m2_measured_force_plot)

        self.puppet_force.append(puppet_measured_force_plot_cat)
        self.a += 1


    def run(self):
        #pdb.set_trace()
        homed_successfully = console.home()
        time.sleep(15)
        
        # initial_position = numpy.array([0, 0, 0.15, 0, 0, 0])
        # arg_initial = self.puppet.move_jp.GetArgumentPrototype()
        # arg_initial.SetGoal(initial_position)
        # self.puppet.move_jp(arg_initial)

        # master_initial_position = numpy.array([0.0, 0.0, 0.0, 0.0, numpy.pi/2, 0.0, 0.0])
        # arg_initial_m1 = self.master1.move_jp.GetArgumentPrototype()
        # arg_initial_m2 = self.master2.move_jp.GetArgumentPrototype()
        # arg_initial_m1.SetGoal(master_initial_position)
        # arg_initial_m2.SetGoal(master_initial_position)
        # self.master1.move_jp(arg_initial_m1)
        # self.master2.move_jp(arg_initial_m2)
        time.sleep(3)
        #teleop_rate = self.ral.create_rate(int(1/self.run_period))
        # print("Running teleop at {} Hz".format(int(1/self.run_period)))
        freq = int(1/self.run_period)
        
        self.enter_aligning()
        print("aligned complete")
        self.running = True


        m1_measured_cp = self.master1.measured_cp()
        m2_measured_cp = self.master2.measured_cp()
        master_1_position = m1_measured_cp.Position()
        master_2_position = m2_measured_cp.Position()
        puppet_measured_cp = self.puppet.measured_cp()
        puppet_measured_pos = puppet_measured_cp.Position()
        self.puppet_measured_rot_fixed = puppet_measured_pos.GetRotation()

        # rot+trans master1
        self.master_1_rotation1_fixed = master_1_position.GetRotation()


        # rot+trans master2
        self.master_2_rotation1_fixed = master_2_position.GetRotation()

        #while not self.ral.is_shutdown():
        # while True:
        while self.a <=18000:
            # check if teleop state should transition
            if self.current_state == teleoperation.State.ALIGNING:
                print("current state transit aligning")
                self.transition_aligning()
            elif self.current_state == teleoperation.State.CLUTCHED:
                print("current state transit clutched")
                self.transition_clutched()
            elif self.current_state == teleoperation.State.FOLLOWING:
                print("current state transit following")
                self.transition_following()
            else:
                raise RuntimeError("Invalid state: {}".format(self.current_state))

            # self.check_arm_state()
            if not self.running:
                break

            # run teleop state handler
            if self.current_state == teleoperation.State.ALIGNING:
                print("current state aligning")
                self.run_aligning()
            elif self.current_state == teleoperation.State.CLUTCHED:
                print("current state clutched")
                self.run_clutched()
            elif self.current_state == teleoperation.State.FOLLOWING:
                print("current state following")
                self.run_following()
            else:
                raise RuntimeError("Invalid state: {}".format(self.current_state))
            
            self.master1_op_state = self.master1.operating_state()
            self.master1_is_busy = self.master1_op_state.GetIsBusy()

            print(f"master1_is_busy : {self.master1_is_busy}")

            time.sleep(self.run_period)

        numpy.savetxt('multi_array_0605.txt', self.y_data_l, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')
        numpy.savetxt('multi_array_exp_0605.txt', self.y_data_l_expected, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')
        numpy.savetxt('multi_m1_force_0605.txt', self.m1_force, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')
        numpy.savetxt('multi_m2_force_0605.txt', self.m2_force, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')
        numpy.savetxt('multi_puppet_force_0605.txt', self.puppet_force, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')
        print(f"run terminated, MTML is busy: {self.master1_is_busy}")


if __name__ == '__main__':
    # parse arguments
    # parser = argparse.ArgumentParser(description = __doc__,
    #                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-m', '--mtm', type = str, required = True,
    #                     choices = ['MTML', 'MTMR'],
    #                     help = 'MTM arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    # args = parser.parse_args(argv)

    parser = argparse.ArgumentParser(description = __doc__,
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alpha', type = float, required = False, default= 0.5,
                         help = 'dominance factor alpha, between 0 and 1')
    parser.add_argument('-b', '--beta', type = float, required = False, default= 0.5,
                         help = 'dominance factor beta, between 0 and 1')
    args = parser.parse_args()

    # ral = crtk.ral('dvrk_python_teleoperation')
    from dvrk_console import *
    # console.power_on()
    #pdb.set_trace()
    mtm1 = MTML
    mtm2 = MTMR
    #mtm2 = MTML2
    psm = PSM1

    clutch = Clutch
    coag = Coag

    application = teleoperation(mtm1, mtm2, psm, clutch, coag, args.alpha, args.beta, 1, 0.001,
                                True, 1)

    application.run()