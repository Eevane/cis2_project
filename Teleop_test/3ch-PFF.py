#!/usr/bin/env python

# Author: Junxiang Wang
# Date: 2024-04-12

# (C) Copyright 2024-2025 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

""" Multilateral teleoperation single system - python wrapper """
""" 4-channel, hub structure """

import os
import argparse
import time
from enum import Enum
import math
import numpy
import sys
import time
from scipy.spatial.transform import Rotation as R
import cisstVectorPython as cisstVector


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
        if operator_present_topic:
            self.operator_is_present = False
            self.operator_button = operator_present_topic
        else:
            self.operator_is_present = True # if not given, then always assume present

        self.clutch_pressed = False
        self.clutch_button = clutch_topic

        self.a = 0
        self.time_data = []
        self.puppet_pos = []
        self.m1_pos = []
        self.m2_pos = []
        self.m1_force = []
        self.m2_force = []
        self.puppet_force = []

        # control law gain
        self.force_gain = 0.35
        self.velocity_gain = 1.0
        self.position_gain = 1.0

    def set_velocity_goal(self, v, base=1.12, max_gain=1.2, threshold=0.02):
        norm = numpy.linalg.norm(v)
        if norm < threshold:
            gain = max_gain
        else:
            gain = base + (max_gain - base) * (threshold / norm)
            gain = min(gain, max_gain)
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

        self.master1.arm.use_gravity_compensation(True)
        self.master2.arm.use_gravity_compensation(True)
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

        self.master1.arm.use_gravity_compensation(True)
        self.master2.arm.use_gravity_compensation(True)

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
        # master1
        master1_measured_cf = self.master1.body_measured_cf()   # (6,) numpy array
        master1_measured_cf[0:3] *= -1.0
        master1_measured_cf[3:6] *= 0   # turn off torque

        # master2
        master2_measured_cf = self.master2.body_measured_cf()   # (6,) numpy array
        master2_measured_cf[0:3] *= -1.0
        master2_measured_cf[3:6] *= 0   # turn off torque

        # puppet
        puppet_measured_cf = self.puppet.body_measured_cf()
        puppet_measured_cf[0:3] *= -1.0
        puppet_measured_cf[3:6] *= 0

        # force input
        gamma = 1.0
        force_goal = self.force_gain * (self.alpha * master1_measured_cf + (1 - self.alpha) * master2_measured_cf + gamma * puppet_measured_cf)


        # Position channel
        master1_measured_trans, master1_measured_rot = self.master1.measured_cp()
        master2_measured_trans, master2_measured_rot = self.master2.measured_cp()
        master1_initial_trans = self.master1_cartesian_initial.GetTranslation()   ####### Reference or copy #########
        master2_initial_trans = self.master2_cartesian_initial.GetTranslation()
        puppet_initial_trans = self.puppet_cartesian_initial.GetTranslation()


        # set translation of psm
        master1_translation = master1_measured_trans - master1_initial_trans
        master2_translation = master2_measured_trans - master2_initial_trans

        master1_translation_scaled = master1_translation
        master2_translation_scaled = master2_translation
        master1_translation_scaled *= self.scale
        master2_translation_scaled *= self.scale
        master_total_translation = self.position_gain * (self.alpha * master1_translation_scaled + (1 - self.alpha) * master2_translation_scaled)
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
        master1_measured_cv_scaled = master1_measured_cv
        master1_measured_cv_scaled[0:3] *= self.velocity_scale      # scale the linear velocity
        master1_measured_cv_scaled[3:6] *= 0.2      # scale down the angular velocity by 0.2

        master2_measured_cv = self.master2.measured_cv()   # master2
        master2_measured_cv_scaled = master2_measured_cv
        master2_measured_cv_scaled[0:3] *= self.velocity_scale     
        master2_measured_cv_scaled[3:6] *= 0.2     

        # average velocity
        puppet_velocity_goal = self.velocity_gain * (self.alpha * master1_measured_cv_scaled + (1 - self.alpha) * master2_measured_cv_scaled)

        # raw_puppet_velocity_goal = self.alpha * master1_measured_cv + (1 - self.alpha) * master2_measured_cv
        # puppet_velocity_goal = self.set_velocity_goal(v=raw_puppet_velocity_goal)

        # Move
        self.puppet.servo_cs(puppet_cartesian_goal, puppet_velocity_goal, force_goal)


        ### Jaw/gripper teleop
        current_master1_gripper = self.master1.gripper_measured_js()
        current_master2_gripper = self.master2.gripper_measured_js()

        master1_ghost_lag = current_master1_gripper - self.gripper_ghost
        master2_ghost_lag = current_master2_gripper - self.gripper_ghost
        # average gripper lag
        ghost_lag = (master1_ghost_lag + master2_ghost_lag) / 2.0

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
        # puppet_translation_scaled = puppet_translation
        # puppet_translation_scaled /= self.scale

        # # set translation of mtm1
        # master1_translation_goal = self.position_gain * puppet_translation_scaled
        # master1_position = master1_translation_goal + master1_initial_trans
        # # set translation of mtm2
        # master2_translation_goal = self.position_gain * puppet_translation_scaled
        # master2_position = master2_translation_goal + master2_initial_trans

        # # set rotation of mtm1
        # master1_rotation = puppet_measured_rot @ numpy.linalg.inv(master1_alignment_offset)
        # # set rotation of mtm2
        # master2_rotation = puppet_measured_rot @ numpy.linalg.inv(master2_alignment_offset)

        # # set cartesian goal of mtm1 and mtm2
        # master1_cartesian_goal = self.set_vctFrm3(rotation=master1_rotation, translation=master1_position)
        # master2_cartesian_goal = self.set_vctFrm3(rotation=master2_rotation, translation=master2_position)


        # # Velocity channel
        # puppet_measured_cv = self.puppet.measured_cv()   # (6,) numpy array
        # puppet_measured_cv_scaled = puppet_measured_cv
        # puppet_measured_cv_scaled[0:3] /= self.velocity_scale      # scale the linear velocity
        # puppet_measured_cv_scaled[3:6] /= 0.2

        # # set velocity goal
        # master1_velocity_goal = self.velocity_gain * puppet_measured_cv_scaled
        # master2_velocity_goal = self.velocity_gain * puppet_measured_cv_scaled

        # # master1_velocity_goal = self.set_velocity_goal(v=puppet_measured_cv)
        # # master2_velocity_goal = self.set_velocity_goal(v=puppet_measured_cv)


        # Move
        self.master1.body_servo_cf(force_goal)
        self.master2.body_servo_cf(force_goal)

        """
        record plotting data
        """
        self.puppet_pos.append(puppet_translation.copy())
        self.m1_pos.append(master1_translation.copy())
        self.m2_pos.append(master2_translation.copy())
        self.m1_force.append(master1_measured_cf.copy())
        self.m2_force.append(master2_measured_cf.copy())
        self.puppet_force.append(puppet_measured_cf.copy())
        # self.a += 1


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

        # puppet_initial_position = numpy.array([-0.00270296, 0.0368143, 0.142947, 1.28645, -0.0889504, 0.174713])
        # self.puppet.move_jp(puppet_initial_position)
        # time.sleep(3)
        
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
                print(f"Time cost relative to {self.run_period} is {to_sleep}")
                time.sleep(to_sleep) if to_sleep > 0 else None
                last_time = time.time()
                
        except KeyboardInterrupt:
            print("Program stopped!")
            system.power_off()

        # save data
        save_path = "/home/xle6/dvrk_teleop_data/Oct_18/3ch_PFF"
        os.makedirs(save_path, exist_ok=True)
        numpy.savetxt(save_path+'psm_pos.txt', self.puppet_pos, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt(save_path+'mtm1_pos.txt', self.m1_pos, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt(save_path+'mtm2_pos.txt', self.m2_pos, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt(save_path+'psm_force.txt', self.puppet_force, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt(save_path+'m1_force.txt', self.m1_force, fmt='%f', delimiter=' ', comments='')
        numpy.savetxt(save_path+'m2_force.txt', self.m2_force, fmt='%f', delimiter=' ', comments='')
        print(f"data.txt saved!")

class ARM:
    def __init__(self, arm, name):
        self.arm = arm
        self.name = name
    
    # def measured_js(self):
    #     measured_js = self.arm.measured_js()
    #     measured_jf = measured_js.Effort()

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
        assert self.name.startswith(("MTML", "MTMR"))
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

    def servo_cp(self, goal):
        assert isinstance(goal, cisstVector.vctFrm3)
        arg = self.arm.move_cp.GetArgumentPrototype()
        arg.SetGoal(goal)
        self.arm.servo_cp(arg)

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
        self.arm.move_cp(arg)

    def move_jp(self, goal):
        arg = self.arm.move_jp.GetArgumentPrototype()
        arg.SetGoal(goal)
        self.arm.move_jp(arg)
        

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alpha', type = float, required = False, default= 0.5,
                         help = 'dominance factor, between 0 and 1')
    parser.add_argument('-n', '--no-mtm-alignment', action='store_true',
                        help="don't align mtm (useful for using haptic devices as MTM which don't have wrist actuation)")
    parser.add_argument('-i', '--interval', type=float, default=0.001,
                        help = 'time interval/period to run at - should be as long as system\'s period to prevent timeouts')
    args = parser.parse_args()

    from dvrk_system import *
    mtm1 = ARM(MTML, 'MTML')
    mtm2 = ARM(MTMR, 'MTMR')
    psm = ARM(PSM1, 'PSM1')

    clutch = clutch
    coag = coag

    application = teleoperation(mtm1, mtm2, psm, clutch, args.interval,
                                not args.no_mtm_alignment, operator_present_topic = coag, alpha = args.alpha)
     
    application.run()
