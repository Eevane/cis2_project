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

import argparse
# import crtk
from enum import Enum
import math
# import std_msgs.msg
import sys
import time
import cisstVectorPython as cisstVector
import pdb

class teleoperation:
    class State(Enum):
        ALIGNING = 1
        CLUTCHED = 2
        FOLLOWING = 3

    def __init__(self, master, puppet, clutch, coag, run_period, align_mtm, operator_present_topic = ""):
        # print('Initialzing dvrk_teleoperation for {} and {}'.format(master.name, puppet.name))
        self.run_period = run_period

        self.master = master
        self.puppet = puppet

        self.clutch = clutch
        self.coag = coag

        self.scale = 0.2
        # self.velocity_scale = self.scale
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

        self.operator_is_active = True
        if operator_present_topic:
            self.operator_is_present = False
        #     self.operator_button = crtk.joystick_button(ral, operator_present_topic)
        #     self.operator_button.set_callback(self.on_operator_present)
        # else:
        #     self.operator_is_present = True # if not given, then always assume present

        self.clutch_pressed = False
        # self.clutch_button = crtk.joystick_button(ral, clutch_topic)
        # self.clutch_button.set_callback(self.on_clutch)

        # for plotting
        self.a = 0
        self.time_data = []
        self.y_data_l = []
        self.y_data_l_expected = []
        self.m1_force = []
        self.m2_force = []
        self.puppet_force = []



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


    # callback for operator pedal/button
    def on_operator_present(self, present):
        self.operator_is_present = present
        if not present:
            self.operator_is_active = False

    # callback for clutch pedal/button
    def on_clutch(self, clutch_pressed):
        self.clutch_pressed = clutch_pressed

    # compute relative orientation of mtm and psm
    def alignment_offset(self):
        # master
        master_measured_cp = self.master.measured_cp()
        master_measured_cp_pos = master_measured_cp.Position()
        master_measured_cp_rot = master_measured_cp_pos.GetRotation()
        # puppet
        puppet_measured_cp = self.puppet.setpoint_cp()
        puppet_measured_cp_pos = puppet_measured_cp.Position()
        puppet_measured_cp_rot = puppet_measured_cp_pos.GetRotation()
        return numpy.linalg.inv(master_measured_cp_rot) @ puppet_measured_cp_rot

    # set relative origins for clutching and alignment offset
    def update_initial_state(self):
        # master
        self.master_cartesian_initial = cisstVector.vctFrm3()
        # measure cp
        measured_cp = self.master.measured_cp()
        measured_cp_pos = measured_cp.Position()
        measured_cp_rot = measured_cp_pos.GetRotation()
        measured_cp_trans = measured_cp_pos.GetTranslation()
        # set
        self.master_cartesian_initial.SetRotation(measured_cp_rot)
        self.master_cartesian_initial.SetTranslation(measured_cp_trans)

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

        self.alignment_offset_initial = self.alignment_offset()
        self.offset_angle, self.offset_axis = self.GetRotAngle(self.alignment_offset_initial)

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
    #     if not self.master.is_homed():
    #         print(f'ERROR: {self.ral.node_name()}: master ({self.master.name}) is not homed anymore')
    #         self.running = False

    def enter_aligning(self):
        self.current_state = teleoperation.State.ALIGNING
        self.last_align = None
        self.last_operator_prompt = time.perf_counter()

        self.master.use_gravity_compensation(True)
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

        alignment_offset = self.alignment_offset()
        orientation_error, _ = self.GetRotAngle(alignment_offset)
        aligned = orientation_error <= self.operator_orientation_tolerance
        if aligned and self.operator_is_active:
            self.enter_following()

    def run_aligning(self):
        orientation_error, _ = self.GetRotAngle(self.alignment_offset())

        # if operator is inactive, use gripper or roll activity to detect when the user is ready
        if self.coag.GetButton():
            gripper_init = self.master.gripper.measured_js()
            gripper = gripper_init.Position()
            self.operator_gripper_max = max(gripper, self.operator_gripper_max)
            self.operator_gripper_min = min(gripper, self.operator_gripper_min)
            gripper_range = self.operator_gripper_max - self.operator_gripper_min
            if gripper_range >= self.operator_gripper_threshold:
                self.operator_is_active = True

            # determine amount of roll around z axis by rotation of y-axis
            measured_cp = self.master.measured_cp()
            setpoint_cp = self.puppet.setpoint_cp()
            master_pos = measured_cp.Position()
            puppet_pos = setpoint_cp.Position()
            master_rotation = master_pos.GetRotation()
            puppet_rotation = puppet_pos.GetRotation()
            master_y_axis = numpy.array([master_rotation[0,1], master_rotation[1,1], master_rotation[2,1]])
            puppet_y_axis = numpy.array([puppet_rotation[0,1], puppet_rotation[1,1], puppet_rotation[2,1]])
            roll = math.acos(numpy.dot(puppet_y_axis, master_y_axis))

            self.operator_roll_max = max(roll, self.operator_roll_max)
            self.operator_roll_min = min(roll, self.operator_roll_min)
            roll_range = self.operator_roll_max - self.operator_roll_min
            if roll_range >= self.operator_roll_threshold:
                self.operator_is_active = True

        # periodically send move_cp to MTM to align with PSM
        aligned = orientation_error <= self.operator_orientation_tolerance
        now = time.perf_counter()
        if not self.last_align or now - self.last_align > 4.0:
            move_cp = cisstVector.vctFrm3()
            measured_cp = self.master.measured_cp()
            setpoint_cp = self.puppet.setpoint_cp()
            master_pos = measured_cp.Position()
            puppet_pos = setpoint_cp.Position()
            master_trans = master_pos.GetTranslation()
            puppet_rotation = puppet_pos.GetRotation()
            move_cp.SetRotation(puppet_rotation)
            move_cp.SetTranslation(master_trans)
            arg = self.master.move_cp.GetArgumentPrototype()
            arg.SetGoal(move_cp)
            self.master.move_cp(arg)
            self.last_align = now

        # periodically notify operator if un-aligned or operator is inactive
        if self.coag.GetButton() and now - self.last_operator_prompt > 4.0:
            self.last_operator_prompt = now
            if not aligned:
                print(f'Unable to align master, angle error is {orientation_error * 180 / math.pi} (deg)')
            elif not self.operator_is_active:
                print(f'To begin teleop, pinch/twist master gripper a bit')

    def enter_clutched(self):
        self.current_state = teleoperation.State.CLUTCHED

        # let MTM position move freely, but lock orientation
        wrench = numpy.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        arg = self.master.body.servo_cf.GetArgumentPrototype()
        arg.SetForce(wrench)
        self.master.body.servo_cf(arg)
        ''' wait for editting'''
        lock_cp = self.master.measured_cp()
        lock_pos = lock_cp.Position()
        lock_rot = lock_pos.GetRotation()
        self.master.lock_orientation(lock_rot)

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

        self.master.use_gravity_compensation(True)

    def transition_following(self):
        if not self.coag.GetButton():
            self.enter_aligning()
        elif self.clutch.GetButton():
            self.enter_clutched()

    def run_following(self):
        ### Cartesian pose teleop
        '''
        Forward
        '''
        
        # Force measurement
        # master
        measured_cf = self.master.body.measured_cf()
        measured_cf_force = measured_cf.Force()
        measured_cf_force[0:3] = measured_cf_force[0:3] * (-1.0)
        measured_cf_force[3:6] = measured_cf_force[3:6] * 0 * 2

        # '''Measure force from joint space'''
        # measured_js = self.master.measured_js()
        # measured_jf = measured_js.Effort()
        # measured_jf[-4:] = 0   # turn off force from the last three axis
        # body_jacobian = self.master.body.jacobian()
        # body_jacobian_trans_inv = numpy.linalg.pinv(body_jacobian.T)
        # measured_cf_force = body_jacobian_trans_inv @ measured_jf
        # measured_cf_force[0:3] = measured_cf_force[0:3] * (-1.0)
        # measured_cf_force[3:6] = measured_cf_force[3:6] * 0 * 2


        # puppet1
        puppet_measured_cf = self.puppet.body.measured_cf()
        puppet_measured_cf_force = puppet_measured_cf.Force()
        puppet_measured_cf_force[0:3] = puppet_measured_cf_force[0:3] * (-1.0)
        puppet_measured_cf_force[3:6] = puppet_measured_cf_force[3:6] * 0 * 2


        # '''Measure force from joint space'''
        # puppet_measured_js = self.puppet.measured_js()
        # puppet_measured_jf = puppet_measured_js.Effort()
        # puppet_measured_jf[-3:] = 0   # turn off force from the last four axis
        # puppet_body_jacobian = self.puppet.body.jacobian()
        # puppet_body_jacobian_trans_inv = numpy.linalg.pinv(puppet_body_jacobian.T)
        # puppet_measured_cf_force = puppet_body_jacobian_trans_inv @ puppet_measured_jf
        # puppet_measured_cf_force[0:3] = puppet_measured_cf_force[0:3] * (1.0)
        # puppet_measured_cf_force[3:6] = puppet_measured_cf_force[3:6] * 0 * 2



        # Force measurement
        force_goal = 0.2 * (measured_cf_force + puppet_measured_cf_force)

        # Position measurement
        #alpha = self.alpha    # position channel dominance factor
        measured_cp = self.master.measured_cp()
        master_position = measured_cp.Position()

        # rot+trans master
        master_rotation1 = master_position.GetRotation()
        master_trans1 = master_position.GetTranslation()
        master_trans2 = self.master_cartesian_initial.GetTranslation()

        master_translation = master_trans1 - master_trans2   # relative translation of master1
        master_puppet_translation = master_translation * self.scale   # convert to puppet frame
        puppet_trans2 = self.puppet_cartesian_initial.GetTranslation()
        master_puppet_translation = master_puppet_translation + puppet_trans2   # translation input of master1 to puppet

        # set rotation of psm to match mtm plus alignment offset
        # if we can actuate the MTM, we slowly reduce the alignment offset to zero over time
        max_delta = self.align_rate * self.run_period
        self.offset_angle += math.copysign(min(abs(self.offset_angle), max_delta), -self.offset_angle)

        # rotation offset master1
        master_alignment_offset = self.GetRotMatrix(self.offset_axis, self.offset_angle)
        master_puppet_rotation = master_rotation1 @ master_alignment_offset

        puppet_rotation = master_puppet_rotation
        puppet_cartesian_goal = cisstVector.vctFrm3()
        puppet_cartesian_goal.SetRotation(puppet_rotation)
        puppet_cartesian_goal.SetTranslation(master_puppet_translation)
        # print(f'puppet_cartesian_goal : {puppet_cartesian_goal}')



        # ##################### use PSM rotation to control PSM
        # # average rotation
        # # puppet_rotation = puppet_measured_rot_fw
        # puppet_rotation = self.puppet_measured_rot_fixed
        # print(f"puppet_rotation : {puppet_rotation}")
        # #####################





        
        # Velocity measurement
        # velocity from master1
        master_measured_cv = self.master.measured_cv()
        master_linear_vel = master_measured_cv.VelocityLinear()
        master_linear_vel = self.velocity_scale * master_linear_vel
        master_angular_vel = master_measured_cv.VelocityAngular()
        vel_fw = numpy.hstack((master_linear_vel, master_angular_vel*0))

        # execute
        arg_fw = self.puppet.servo_cs.GetArgumentPrototype()
        arg_fw.SetPositionIsValid(True)
        arg_fw.SetPosition(puppet_cartesian_goal)
        #print(f'master_cartesian_goal: {master_cartesian_goal}')
        arg_fw.SetVelocityIsValid(True)
        arg_fw.SetVelocity(vel_fw)
        #print(f'vel_cs : {vel_cs}')
        arg_fw.SetForceIsValid(True)
        arg_fw.SetForce(force_goal)
        #print(f'force_PSM_cs : {force_PSM_cs}')

        self.puppet.servo_cs(arg_fw)


        ### Jaw/gripper teleop --- so far only master1 can control the jaw
        # master 1
        master_gripper_measured_js_init = self.master.gripper.measured_js()
        master_current_gripper = master_gripper_measured_js_init.Position()
        master_ghost_lag = master_current_gripper - self.gripper_ghost

        # average
        average_ghost_lag = master_ghost_lag

        max_delta = self.jaw_rate * self.run_period
        # move ghost at most max_delta towards current gripper
        self.gripper_ghost += math.copysign(min(abs(average_ghost_lag), max_delta), average_ghost_lag)
        
        # gripper_to_jaw = self.gripper_to_jaw(self.gripper_ghost)
        arg = self.puppet.jaw.servo_jp.GetArgumentPrototype()
        arg.SetGoal(numpy.array([self.gripper_to_jaw(self.gripper_ghost)]))
        self.puppet.jaw.servo_jp(arg)
        #print('self.puppet.servo_jp(arg)')

        '''
        Backward
        '''
        # Position measurement (only measure puppet's)
        puppet_measured_cp = self.puppet.measured_cp()
        puppet_measured_pos = puppet_measured_cp.Position()
        puppet_measured_rot = puppet_measured_pos.GetRotation()
        puppet_measured_trans = puppet_measured_pos.GetTranslation()

        puppet_relative_translation = puppet_measured_trans - self.puppet_cartesian_initial.GetTranslation()
        master_relative_translation = puppet_relative_translation / self.scale

        # relative trans
        translation_cs = master_relative_translation + self.master_cartesian_initial.GetTranslation()

        # relative rot
        rotation_cs = puppet_measured_rot @ numpy.linalg.inv(master_alignment_offset)

        # set
        # master
        cartesian_goal = cisstVector.vctFrm3()
        cartesian_goal.SetRotation(rotation_cs)
        cartesian_goal.SetTranslation(translation_cs)

        # Velocity measurement (only measure puppet's)
        puppet_measured_cv = self.puppet.measured_cv()
        linear_vel_cs = puppet_measured_cv.VelocityLinear()
        linear_vel_cs = (1/self.velocity_scale) * linear_vel_cs
        angular_vel_cs = puppet_measured_cv.VelocityAngular()
        angular_vel_cs = angular_vel_cs  * 0.0   # zero angular velocity
        
        vel_cs = numpy.hstack((linear_vel_cs, angular_vel_cs))


        # master1 arg
        arg = self.master.servo_cs.GetArgumentPrototype()
        arg.SetPositionIsValid(True)
        arg.SetPosition(cartesian_goal)
        # print(f'master_cartesian_goal: {m1_cartesian_goal}')
        arg.SetVelocityIsValid(True)
        arg.SetVelocity(vel_cs)
        # print(f'vel_cs : {vel_cs}')
        arg.SetForceIsValid(True)
        arg.SetForce(force_goal)
        self.master.servo_cs(arg)

        # measure master1 force for plot
        m1_measured_cf_plot = self.master.body.measured_cf()
        m1_measured_force_plot = m1_measured_cf_plot.Force()
        m1_measured_force_plot = m1_measured_force_plot[0:3] * (1)

        # '''
        # plot
        # '''
        puppet_measured_cp_plot = self.puppet.measured_cp()
        puppet_measured_pos_plot = puppet_measured_cp_plot.Position()
        puppet_measured_trans_plot = puppet_measured_pos_plot.GetTranslation()

        # measure puppet force for plot
        puppet_measured_cf_plot = self.puppet.body.measured_cf()
        puppet_measured_force_plot = puppet_measured_cf_plot.Force()
        puppet_measured_force_plot_cat = puppet_measured_force_plot[0:3] * 1
        #print(f"arg_fw : {arg_fw}")

        self.y_data_l.append(puppet_measured_trans_plot)
        self.y_data_l_expected.append(master_puppet_translation)

        self.m1_force.append(m1_measured_force_plot)

        self.puppet_force.append(puppet_measured_force_plot_cat)
        self.a += 1



    # def home(self):
    #     print("Homing arms...")
    #     timeout = 10.0 # seconds
    #     if not self.puppet.enable(timeout) or not self.puppet.home(timeout):
    #         print('    ! failed to home {} within {} seconds'.format(self.puppet.name, timeout))
    #         return False

    #     if not self.master.enable(timeout) or not self.master.home(timeout):
    #         print('    ! failed to home {} within {} seconds'.format(self.master.name, timeout))
    #         return False

    #     print("    Homing is complete")
    #     return True

    def run(self):
        #pdb.set_trace()
        homed_successfully = console.home()
        time.sleep(10)
        print("home complete")
        if not homed_successfully:
            print("home not success")
            return

        # initial_position = numpy.array([0, 0, 0.13, 0, 0, 0])
        # arg_initial = self.puppet.move_jp.GetArgumentPrototype()
        # arg_initial.SetGoal(initial_position)
        # self.puppet.move_jp(arg_initial)
        # time.sleep(3)

        # puppet_measured_cp = self.puppet.measured_cp()
        # puppet_measured_pos = puppet_measured_cp.Position()
        # self.puppet_measured_rot_fixed = puppet_measured_pos.GetRotation()

        self.enter_aligning()
        print("aligned complete")
        self.running = True


        '''# fix orientation
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
        '''

        self.enter_aligning()
        print("aligned complete")
        self.running = True

        #while not self.ral.is_shutdown():
        # while True:
        while self.a <=18000:
            # check if teleop state should transition
            if self.current_state == teleoperation.State.ALIGNING:
                #print("current state transit aligning")
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
                #print("current state aligning")
                self.run_aligning()
            elif self.current_state == teleoperation.State.CLUTCHED:
                print("current state clutched")
                self.run_clutched()
            elif self.current_state == teleoperation.State.FOLLOWING:
                print("current state following")
                self.run_following()
            else:
                raise RuntimeError("Invalid state: {}".format(self.current_state))

            time.sleep(self.run_period)


        numpy.savetxt('bi_array_0605_1.txt', self.y_data_l, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')
        numpy.savetxt('bi_array_exp_0605_1.txt', self.y_data_l_expected, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')
        numpy.savetxt('bi_m1_force_0605_1.txt', self.m1_force, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')
        numpy.savetxt('bi_puppet_force_0605_1.txt', self.puppet_force, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')
        print(f"Program finished!")

if __name__ == '__main__':
    # extract ros arguments (e.g. __ns:= for namespace)
    # argv = crtk.ral.parse_argv(sys.argv[1:]) # skip argv[0], script name

    # parse arguments
    # parser = argparse.ArgumentParser(description = __doc__,
    #                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-m', '--mtm', type = str, required = True,
    #                     choices = ['MTML', 'MTMR'],
    #                     help = 'MTM arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    # parser.add_argument('-p', '--psm', type = str, required = True,
    #                     choices = ['PSM1', 'PSM2', 'PSM3'],
    #                     help = 'PSM arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    # parser.add_argument('-c', '--clutch', type = str, default='/footpedals/clutch',
    #                     help = 'ROS topic corresponding to clutch button/pedal input')
    # parser.add_argument('-o', '--operator', type = str, default='/footpedals/coag', const=None, nargs='?',
    #                     help = 'ROS topic corresponding to operator present button/pedal/sensor input - use "-o" without an argument to disable')
    # parser.add_argument('-n', '--no-mtm-alignment', action='store_true',
    #                     help="don't align mtm (useful for using haptic devices as MTM which don't have wrist actuation)")
    # parser.add_argument('-i', '--interval', type=float, default=0.005,
    #                     help = 'time interval/period to run at - should be as long as console\'s period to prevent timeouts')
    # args = parser.parse_args(argv)

    # ral = crtk.ral('dvrk_python_teleoperation')
    from dvrk_console import *
    console.power_on()
    #pdb.set_trace()
    mtm = MTML
    psm = PSM1

    clutch = Clutch
    coag = Coag
    application = teleoperation(mtm, psm, clutch, coag, 0.001,
                                True, 1)
    application.run()